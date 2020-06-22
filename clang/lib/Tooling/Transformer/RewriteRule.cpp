//===--- Transformer.cpp - Transformer library implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace transformer;

using ast_matchers::MatchFinder;
using ast_matchers::internal::DynTypedMatcher;

using MatchResult = MatchFinder::MatchResult;

static Expected<SmallVector<transformer::Edit, 1>>
translateEdits(const MatchResult &Result, ArrayRef<ASTEdit> ASTEdits) {
  SmallVector<transformer::Edit, 1> Edits;
  for (const auto &E : ASTEdits) {
    Expected<CharSourceRange> Range = E.TargetRange(Result);
    if (!Range)
      return Range.takeError();
    llvm::Optional<CharSourceRange> EditRange =
        tooling::getRangeForEdit(*Range, *Result.Context);
    // FIXME: let user specify whether to treat this case as an error or ignore
    // it as is currently done.
    if (!EditRange)
      return SmallVector<Edit, 0>();
    auto Replacement = E.Replacement->eval(Result);
    if (!Replacement)
      return Replacement.takeError();
    transformer::Edit T;
    T.Range = *EditRange;
    T.Replacement = std::move(*Replacement);
    Edits.push_back(std::move(T));
  }
  return Edits;
}

EditGenerator transformer::editList(SmallVector<ASTEdit, 1> Edits) {
  return [Edits = std::move(Edits)](const MatchResult &Result) {
    return translateEdits(Result, Edits);
  };
}

EditGenerator transformer::edit(ASTEdit Edit) {
  return [Edit = std::move(Edit)](const MatchResult &Result) {
    return translateEdits(Result, {Edit});
  };
}

ASTEdit transformer::changeTo(RangeSelector Target, TextGenerator Replacement) {
  ASTEdit E;
  E.TargetRange = std::move(Target);
  E.Replacement = std::move(Replacement);
  return E;
}

namespace {
/// A \c TextGenerator that always returns a fixed string.
class SimpleTextGenerator : public MatchComputation<std::string> {
  std::string S;

public:
  SimpleTextGenerator(std::string S) : S(std::move(S)) {}
  llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &,
                   std::string *Result) const override {
    Result->append(S);
    return llvm::Error::success();
  }
  std::string toString() const override {
    return (llvm::Twine("text(\"") + S + "\")").str();
  }
};
} // namespace

ASTEdit transformer::remove(RangeSelector S) {
  return change(std::move(S), std::make_shared<SimpleTextGenerator>(""));
}

RewriteRule transformer::makeRule(ast_matchers::internal::DynTypedMatcher M,
                                  EditGenerator Edits,
                                  TextGenerator Explanation) {
  return RewriteRule{{RewriteRule::Case{
      std::move(M), std::move(Edits), std::move(Explanation), {}}}};
}

void transformer::addInclude(RewriteRule &Rule, StringRef Header,
                         IncludeFormat Format) {
  for (auto &Case : Rule.Cases)
    Case.AddedIncludes.emplace_back(Header.str(), Format);
}

#ifndef NDEBUG
// Filters for supported matcher kinds. FIXME: Explicitly list the allowed kinds
// (all node matcher types except for `QualType` and `Type`), rather than just
// banning `QualType` and `Type`.
static bool hasValidKind(const DynTypedMatcher &M) {
  return !M.canConvertTo<QualType>();
}
#endif

// Binds each rule's matcher to a unique (and deterministic) tag based on
// `TagBase` and the id paired with the case. All of the returned matchers have
// their traversal kind explicitly set, either based on a pre-set kind or to the
// provided `DefaultTraversalKind`.
static std::vector<DynTypedMatcher> taggedMatchers(
    StringRef TagBase,
    const SmallVectorImpl<std::pair<size_t, RewriteRule::Case>> &Cases,
    ast_type_traits::TraversalKind DefaultTraversalKind) {
  std::vector<DynTypedMatcher> Matchers;
  Matchers.reserve(Cases.size());
  for (const auto &Case : Cases) {
    std::string Tag = (TagBase + Twine(Case.first)).str();
    // HACK: Many matchers are not bindable, so ensure that tryBind will work.
    DynTypedMatcher BoundMatcher(Case.second.Matcher);
    BoundMatcher.setAllowBind(true);
    auto M = *BoundMatcher.tryBind(Tag);
    Matchers.push_back(!M.getTraversalKind()
                           ? M.withTraversalKind(DefaultTraversalKind)
                           : std::move(M));
  }
  return Matchers;
}

// Simply gathers the contents of the various rules into a single rule. The
// actual work to combine these into an ordered choice is deferred to matcher
// registration.
RewriteRule transformer::applyFirst(ArrayRef<RewriteRule> Rules) {
  RewriteRule R;
  for (auto &Rule : Rules)
    R.Cases.append(Rule.Cases.begin(), Rule.Cases.end());
  return R;
}

std::vector<DynTypedMatcher>
transformer::detail::buildMatchers(const RewriteRule &Rule) {
  // Map the cases into buckets of matchers -- one for each "root" AST kind,
  // which guarantees that they can be combined in a single anyOf matcher. Each
  // case is paired with an identifying number that is converted to a string id
  // in `taggedMatchers`.
  std::map<ASTNodeKind, SmallVector<std::pair<size_t, RewriteRule::Case>, 1>>
      Buckets;
  const SmallVectorImpl<RewriteRule::Case> &Cases = Rule.Cases;
  for (int I = 0, N = Cases.size(); I < N; ++I) {
    assert(hasValidKind(Cases[I].Matcher) &&
           "Matcher must be non-(Qual)Type node matcher");
    Buckets[Cases[I].Matcher.getSupportedKind()].emplace_back(I, Cases[I]);
  }

  // Each anyOf explicitly controls the traversal kind. The anyOf itself is set
  // to `TK_AsIs` to ensure no nodes are skipped, thereby deferring to the kind
  // of the branches. Then, each branch is either left as is, if the kind is
  // already set, or explicitly set to `TK_IgnoreUnlessSpelledInSource`. We
  // choose this setting, because we think it is the one most friendly to
  // beginners, who are (largely) the target audience of Transformer.
  std::vector<DynTypedMatcher> Matchers;
  for (const auto &Bucket : Buckets) {
    DynTypedMatcher M = DynTypedMatcher::constructVariadic(
        DynTypedMatcher::VO_AnyOf, Bucket.first,
        taggedMatchers("Tag", Bucket.second, TK_IgnoreUnlessSpelledInSource));
    M.setAllowBind(true);
    // `tryBind` is guaranteed to succeed, because `AllowBind` was set to true.
    Matchers.push_back(
        M.tryBind(RewriteRule::RootID)->withTraversalKind(TK_AsIs));
  }
  return Matchers;
}

DynTypedMatcher transformer::detail::buildMatcher(const RewriteRule &Rule) {
  std::vector<DynTypedMatcher> Ms = buildMatchers(Rule);
  assert(Ms.size() == 1 && "Cases must have compatible matchers.");
  return Ms[0];
}

SourceLocation transformer::detail::getRuleMatchLoc(const MatchResult &Result) {
  auto &NodesMap = Result.Nodes.getMap();
  auto Root = NodesMap.find(RewriteRule::RootID);
  assert(Root != NodesMap.end() && "Transformation failed: missing root node.");
  llvm::Optional<CharSourceRange> RootRange = tooling::getRangeForEdit(
      CharSourceRange::getTokenRange(Root->second.getSourceRange()),
      *Result.Context);
  if (RootRange)
    return RootRange->getBegin();
  // The match doesn't have a coherent range, so fall back to the expansion
  // location as the "beginning" of the match.
  return Result.SourceManager->getExpansionLoc(
      Root->second.getSourceRange().getBegin());
}

// Finds the case that was "selected" -- that is, whose matcher triggered the
// `MatchResult`.
const RewriteRule::Case &
transformer::detail::findSelectedCase(const MatchResult &Result,
                                  const RewriteRule &Rule) {
  if (Rule.Cases.size() == 1)
    return Rule.Cases[0];

  auto &NodesMap = Result.Nodes.getMap();
  for (size_t i = 0, N = Rule.Cases.size(); i < N; ++i) {
    std::string Tag = ("Tag" + Twine(i)).str();
    if (NodesMap.find(Tag) != NodesMap.end())
      return Rule.Cases[i];
  }
  llvm_unreachable("No tag found for this rule.");
}

constexpr llvm::StringLiteral RewriteRule::RootID;

TextGenerator tooling::text(std::string M) {
  return std::make_shared<SimpleTextGenerator>(std::move(M));
}
