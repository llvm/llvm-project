//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionEscapeCheck.h"

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {
namespace {

AST_MATCHER_P(FunctionDecl, isEnabled, llvm::StringSet<>,
              FunctionsThatShouldNotThrow) {
  return FunctionsThatShouldNotThrow.contains(Node.getNameAsString());
}

AST_MATCHER(FunctionDecl, isExplicitThrow) {
  return isExplicitThrowExceptionSpec(Node.getExceptionSpecType()) &&
         Node.getExceptionSpecSourceRange().isValid();
}

AST_MATCHER(FunctionDecl, hasAtLeastOneParameter) {
  return Node.getNumParams() > 0;
}

} // namespace

ExceptionEscapeCheck::ExceptionEscapeCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), RawFunctionsThatShouldNotThrow(Options.get(
                                         "FunctionsThatShouldNotThrow", "")),
      RawIgnoredExceptions(Options.get("IgnoredExceptions", "")),
      RawCheckedSwapFunctions(
          Options.get("CheckedSwapFunctions", "swap,iter_swap,iter_move")),
      CheckDestructors(Options.get("CheckDestructors", true)),
      CheckMoveMemberFunctions(Options.get("CheckMoveMemberFunctions", true)),
      CheckMain(Options.get("CheckMain", true)),
      CheckNothrowFunctions(Options.get("CheckNothrowFunctions", true)),
      KnownUnannotatedAsThrowing(
          Options.get("KnownUnannotatedAsThrowing", false)),
      UnknownAsThrowing(Options.get("UnknownAsThrowing", false)) {
  llvm::SmallVector<StringRef, 8> FunctionsThatShouldNotThrowVec,
      IgnoredExceptionsVec, CheckedSwapFunctionsVec;
  RawFunctionsThatShouldNotThrow.split(FunctionsThatShouldNotThrowVec, ",", -1,
                                       false);
  FunctionsThatShouldNotThrow.insert_range(FunctionsThatShouldNotThrowVec);

  RawCheckedSwapFunctions.split(CheckedSwapFunctionsVec, ",", -1, false);
  CheckedSwapFunctions.insert_range(CheckedSwapFunctionsVec);

  llvm::StringSet<> IgnoredExceptions;
  RawIgnoredExceptions.split(IgnoredExceptionsVec, ",", -1, false);
  IgnoredExceptions.insert_range(IgnoredExceptionsVec);
  Tracer.ignoreExceptions(std::move(IgnoredExceptions));
  Tracer.ignoreBadAlloc(true);
  Tracer.assumeUnannotatedFunctionsThrow(KnownUnannotatedAsThrowing);
}

void ExceptionEscapeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionsThatShouldNotThrow",
                RawFunctionsThatShouldNotThrow);
  Options.store(Opts, "IgnoredExceptions", RawIgnoredExceptions);
  Options.store(Opts, "CheckedSwapFunctions", RawCheckedSwapFunctions);
  Options.store(Opts, "CheckDestructors", CheckDestructors);
  Options.store(Opts, "CheckMoveMemberFunctions", CheckMoveMemberFunctions);
  Options.store(Opts, "CheckMain", CheckMain);
  Options.store(Opts, "CheckNothrowFunctions", CheckNothrowFunctions);
  Options.store(Opts, "KnownUnannotatedAsThrowing", KnownUnannotatedAsThrowing);
  Options.store(Opts, "UnknownAsThrowing", UnknownAsThrowing);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  auto MatchIf = [](bool Enabled, const auto &Matcher) {
    ast_matchers::internal::Matcher<FunctionDecl> Nothing = unless(anything());
    return Enabled ? Matcher : Nothing;
  };
  Finder->addMatcher(
      functionDecl(
          isDefinition(),
          anyOf(
              MatchIf(CheckNothrowFunctions, isNoThrow()),
              allOf(anyOf(MatchIf(CheckDestructors, cxxDestructorDecl()),
                          MatchIf(
                              CheckMoveMemberFunctions,
                              anyOf(cxxConstructorDecl(isMoveConstructor()),
                                    cxxMethodDecl(isMoveAssignmentOperator()))),
                          MatchIf(CheckMain, isMain()),
                          allOf(isEnabled(CheckedSwapFunctions),
                                hasAtLeastOneParameter())),
                    unless(isExplicitThrow())),
              isEnabled(FunctionsThatShouldNotThrow)))
          .bind("thrower"),
      this);
}

void ExceptionEscapeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("thrower");

  if (!MatchedDecl)
    return;

  const utils::ExceptionAnalyzer::ExceptionInfo Info =
      Tracer.analyze(MatchedDecl);

  const auto Behaviour = Info.getBehaviour();
  const bool IsThrowing =
      Behaviour == utils::ExceptionAnalyzer::State::Throwing;
  const bool IsUnknown = Behaviour == utils::ExceptionAnalyzer::State::Unknown;

  const bool ReportUnknown =
      IsUnknown &&
      ((KnownUnannotatedAsThrowing && Info.hasUnknownFromKnownUnannotated()) ||
       (UnknownAsThrowing && Info.hasUnknownFromMissingDefinition()));

  if (!(IsThrowing || ReportUnknown))
    return;

  diag(MatchedDecl->getLocation(), "an exception may be thrown in function %0 "
                                   "which should not throw exceptions")
      << MatchedDecl;

  if (Info.getExceptions().empty())
    return;

  const auto &[ThrowType, ThrowInfo] = *Info.getExceptions().begin();

  if (ThrowInfo.Loc.isInvalid())
    return;

  const utils::ExceptionAnalyzer::CallStack &Stack = ThrowInfo.Stack;
  diag(ThrowInfo.Loc,
       "frame #0: unhandled exception of type %0 may be thrown in function "
       "%1 here",
       DiagnosticIDs::Note)
      << QualType(ThrowType, 0U) << Stack.back().first;

  size_t FrameNo = 1;
  for (auto CurrIt = ++Stack.rbegin(), PrevIt = Stack.rbegin();
       CurrIt != Stack.rend(); ++CurrIt, ++PrevIt) {
    const FunctionDecl *CurrFunction = CurrIt->first;
    const FunctionDecl *PrevFunction = PrevIt->first;
    const SourceLocation PrevLocation = PrevIt->second;
    if (PrevLocation.isValid()) {
      diag(PrevLocation, "frame #%0: function %1 calls function %2 here",
           DiagnosticIDs::Note)
          << FrameNo << CurrFunction << PrevFunction;
    } else {
      diag(CurrFunction->getLocation(),
           "frame #%0: function %1 calls function %2", DiagnosticIDs::Note)
          << FrameNo << CurrFunction << PrevFunction;
    }
    ++FrameNo;
  }
}

} // namespace clang::tidy::bugprone
