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
#include <deque>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace tooling;

using ast_matchers::MatchFinder;
using ast_matchers::internal::DynTypedMatcher;
using ast_type_traits::ASTNodeKind;
using ast_type_traits::DynTypedNode;
using llvm::Error;
using llvm::StringError;

using MatchResult = MatchFinder::MatchResult;

// Did the text at this location originate in a macro definition (aka. body)?
// For example,
//
//   #define NESTED(x) x
//   #define MACRO(y) { int y  = NESTED(3); }
//   if (true) MACRO(foo)
//
// The if statement expands to
//
//   if (true) { int foo = 3; }
//                   ^     ^
//                   Loc1  Loc2
//
// For SourceManager SM, SM.isMacroArgExpansion(Loc1) and
// SM.isMacroArgExpansion(Loc2) are both true, but isOriginMacroBody(sm, Loc1)
// is false, because "foo" originated in the source file (as an argument to a
// macro), whereas isOriginMacroBody(SM, Loc2) is true, because "3" originated
// in the definition of MACRO.
static bool isOriginMacroBody(const clang::SourceManager &SM,
                              clang::SourceLocation Loc) {
  while (Loc.isMacroID()) {
    if (SM.isMacroBodyExpansion(Loc))
      return true;
    // Otherwise, it must be in an argument, so we continue searching up the
    // invocation stack. getImmediateMacroCallerLoc() gives the location of the
    // argument text, inside the call text.
    Loc = SM.getImmediateMacroCallerLoc(Loc);
  }
  return false;
}

Expected<SmallVector<tooling::detail::Transformation, 1>>
tooling::detail::translateEdits(const MatchResult &Result,
                                llvm::ArrayRef<ASTEdit> Edits) {
  SmallVector<tooling::detail::Transformation, 1> Transformations;
  for (const auto &Edit : Edits) {
    Expected<CharSourceRange> Range = Edit.TargetRange(Result);
    if (!Range)
      return Range.takeError();
    if (Range->isInvalid() ||
        isOriginMacroBody(*Result.SourceManager, Range->getBegin()))
      return SmallVector<Transformation, 0>();
    auto Replacement = Edit.Replacement(Result);
    if (!Replacement)
      return Replacement.takeError();
    tooling::detail::Transformation T;
    T.Range = *Range;
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

// Determines whether A is a base type of B in the class hierarchy, including
// the implicit relationship of Type and QualType.
static bool isBaseOf(ASTNodeKind A, ASTNodeKind B) {
  static auto TypeKind = ASTNodeKind::getFromNodeKind<Type>();
  static auto QualKind = ASTNodeKind::getFromNodeKind<QualType>();
  /// Mimic the implicit conversions of Matcher<>.
  /// - From Matcher<Type> to Matcher<QualType>
  /// - From Matcher<Base> to Matcher<Derived>
  return (A.isSame(TypeKind) && B.isSame(QualKind)) || A.isBaseOf(B);
}

// Try to find a common kind to which all of the rule's matchers can be
// converted.
static ASTNodeKind
findCommonKind(const SmallVectorImpl<RewriteRule::Case> &Cases) {
  assert(!Cases.empty() && "Rule must have at least one case.");
  ASTNodeKind JoinKind = Cases[0].Matcher.getSupportedKind();
  // Find a (least) Kind K, for which M.canConvertTo(K) holds, for all matchers
  // M in Rules.
  for (const auto &Case : Cases) {
    auto K = Case.Matcher.getSupportedKind();
    if (isBaseOf(JoinKind, K)) {
      JoinKind = K;
      continue;
    }
    if (K.isSame(JoinKind) || isBaseOf(K, JoinKind))
      // JoinKind is already the lowest.
      continue;
    // K and JoinKind are unrelated -- there is no least common kind.
    return ASTNodeKind();
  }
  return JoinKind;
}

// Binds each rule's matcher to a unique (and deterministic) tag based on
// `TagBase`.
static std::vector<DynTypedMatcher>
taggedMatchers(StringRef TagBase,
               const SmallVectorImpl<RewriteRule::Case> &Cases) {
  std::vector<DynTypedMatcher> Matchers;
  Matchers.reserve(Cases.size());
  size_t count = 0;
  for (const auto &Case : Cases) {
    std::string Tag = (TagBase + Twine(count)).str();
    ++count;
    auto M = Case.Matcher.tryBind(Tag);
    assert(M && "RewriteRule matchers should be bindable.");
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

static DynTypedMatcher joinCaseMatchers(const RewriteRule &Rule) {
  assert(!Rule.Cases.empty() && "Rule must have at least one case.");
  if (Rule.Cases.size() == 1)
    return Rule.Cases[0].Matcher;

  auto CommonKind = findCommonKind(Rule.Cases);
  assert(!CommonKind.isNone() && "Cases must have compatible matchers.");
  return DynTypedMatcher::constructVariadic(
      DynTypedMatcher::VO_AnyOf, CommonKind, taggedMatchers("Tag", Rule.Cases));
}

DynTypedMatcher tooling::detail::buildMatcher(const RewriteRule &Rule) {
  DynTypedMatcher M = joinCaseMatchers(Rule);
  M.setAllowBind(true);
  // `tryBind` is guaranteed to succeed, because `AllowBind` was set to true.
  return *M.tryBind(RewriteRule::RootID);
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
  MatchFinder->addDynamicMatcher(tooling::detail::buildMatcher(Rule), this);
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
