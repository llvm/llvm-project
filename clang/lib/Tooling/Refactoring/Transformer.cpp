//===--- Transformer.cpp - Transformer library implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactoring/Transformer.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/Refactoring/AtomicChange.h"
#include "clang/Tooling/Refactoring/SourceCode.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include <string>
#include <utility>
#include <vector>
#include <map>

using namespace clang;
using namespace tooling;

using ast_matchers::MatchFinder;
using ast_matchers::internal::DynTypedMatcher;
using ast_type_traits::ASTNodeKind;
using ast_type_traits::DynTypedNode;
using llvm::Error;
using llvm::StringError;

using MatchResult = MatchFinder::MatchResult;

Expected<SmallVector<tooling::detail::Transformation, 1>>
tooling::detail::translateEdits(const MatchResult &Result,
                                llvm::ArrayRef<ASTEdit> Edits) {
  SmallVector<tooling::detail::Transformation, 1> Transformations;
  for (const auto &Edit : Edits) {
    Expected<CharSourceRange> Range = Edit.TargetRange(Result);
    if (!Range)
      return Range.takeError();
    llvm::Optional<CharSourceRange> EditRange =
        getRangeForEdit(*Range, *Result.Context);
    // FIXME: let user specify whether to treat this case as an error or ignore
    // it as is currently done.
    if (!EditRange)
      return SmallVector<Transformation, 0>();
    auto Replacement = Edit.Replacement(Result);
    if (!Replacement)
      return Replacement.takeError();
    tooling::detail::Transformation T;
    T.Range = *EditRange;
    T.Replacement = std::move(*Replacement);
    Transformations.push_back(std::move(T));
  }
  return Transformations;
}

ASTEdit tooling::change(RangeSelector S, TextGenerator Replacement) {
  ASTEdit E;
  E.TargetRange = std::move(S);
  E.Replacement = std::move(Replacement);
  return E;
}

RewriteRule tooling::makeRule(DynTypedMatcher M, SmallVector<ASTEdit, 1> Edits,
                              TextGenerator Explanation) {
  return RewriteRule{{RewriteRule::Case{
      std::move(M), std::move(Edits), std::move(Explanation), {}}}};
}

void tooling::addInclude(RewriteRule &Rule, StringRef Header,
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
// `TagBase` and the id paired with the case.
static std::vector<DynTypedMatcher> taggedMatchers(
    StringRef TagBase,
    const SmallVectorImpl<std::pair<size_t, RewriteRule::Case>> &Cases) {
  std::vector<DynTypedMatcher> Matchers;
  Matchers.reserve(Cases.size());
  for (const auto &Case : Cases) {
    std::string Tag = (TagBase + Twine(Case.first)).str();
    // HACK: Many matchers are not bindable, so ensure that tryBind will work.
    DynTypedMatcher BoundMatcher(Case.second.Matcher);
    BoundMatcher.setAllowBind(true);
    auto M = BoundMatcher.tryBind(Tag);
    Matchers.push_back(*std::move(M));
  }
  return Matchers;
}

// Simply gathers the contents of the various rules into a single rule. The
// actual work to combine these into an ordered choice is deferred to matcher
// registration.
RewriteRule tooling::applyFirst(ArrayRef<RewriteRule> Rules) {
  RewriteRule R;
  for (auto &Rule : Rules)
    R.Cases.append(Rule.Cases.begin(), Rule.Cases.end());
  return R;
}

std::vector<DynTypedMatcher>
tooling::detail::buildMatchers(const RewriteRule &Rule) {
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

  std::vector<DynTypedMatcher> Matchers;
  for (const auto &Bucket : Buckets) {
    DynTypedMatcher M = DynTypedMatcher::constructVariadic(
        DynTypedMatcher::VO_AnyOf, Bucket.first,
        taggedMatchers("Tag", Bucket.second));
    M.setAllowBind(true);
    // `tryBind` is guaranteed to succeed, because `AllowBind` was set to true.
    Matchers.push_back(*M.tryBind(RewriteRule::RootID));
  }
  return Matchers;
}

DynTypedMatcher tooling::detail::buildMatcher(const RewriteRule &Rule) {
  std::vector<DynTypedMatcher> Ms = buildMatchers(Rule);
  assert(Ms.size() == 1 && "Cases must have compatible matchers.");
  return Ms[0];
}

// Finds the case that was "selected" -- that is, whose matcher triggered the
// `MatchResult`.
const RewriteRule::Case &
tooling::detail::findSelectedCase(const MatchResult &Result,
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

void Transformer::registerMatchers(MatchFinder *MatchFinder) {
  for (auto &Matcher : tooling::detail::buildMatchers(Rule))
    MatchFinder->addDynamicMatcher(Matcher, this);
}

void Transformer::run(const MatchResult &Result) {
  if (Result.Context->getDiagnostics().hasErrorOccurred())
    return;

  // Verify the existence and validity of the AST node that roots this rule.
  auto &NodesMap = Result.Nodes.getMap();
  auto Root = NodesMap.find(RewriteRule::RootID);
  assert(Root != NodesMap.end() && "Transformation failed: missing root node.");
  SourceLocation RootLoc = Result.SourceManager->getExpansionLoc(
      Root->second.getSourceRange().getBegin());
  assert(RootLoc.isValid() && "Invalid location for Root node of match.");

  RewriteRule::Case Case = tooling::detail::findSelectedCase(Result, Rule);
  auto Transformations = tooling::detail::translateEdits(Result, Case.Edits);
  if (!Transformations) {
    Consumer(Transformations.takeError());
    return;
  }

  if (Transformations->empty()) {
    // No rewrite applied (but no error encountered either).
    RootLoc.print(llvm::errs() << "note: skipping match at loc ",
                  *Result.SourceManager);
    llvm::errs() << "\n";
    return;
  }

  // Record the results in the AtomicChange.
  AtomicChange AC(*Result.SourceManager, RootLoc);
  for (const auto &T : *Transformations) {
    if (auto Err = AC.replace(*Result.SourceManager, T.Range, T.Replacement)) {
      Consumer(std::move(Err));
      return;
    }
  }

  for (const auto &I : Case.AddedIncludes) {
    auto &Header = I.first;
    switch (I.second) {
    case IncludeFormat::Quoted:
      AC.addHeader(Header);
      break;
    case IncludeFormat::Angled:
      AC.addHeader((llvm::Twine("<") + Header + ">").str());
      break;
    }
  }

  Consumer(std::move(AC));
}
