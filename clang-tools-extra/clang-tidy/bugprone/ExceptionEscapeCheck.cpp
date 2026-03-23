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

namespace clang::tidy {

template <>
struct OptionEnumMapping<
    bugprone::ExceptionEscapeCheck::TreatFunctionsWithoutSpecification> {
  using TreatFunctionsWithoutSpecification =
      bugprone::ExceptionEscapeCheck::TreatFunctionsWithoutSpecification;

  static llvm::ArrayRef<
      std::pair<TreatFunctionsWithoutSpecification, StringRef>>
  getEnumMapping() {
    static constexpr std::pair<TreatFunctionsWithoutSpecification, StringRef>
        Mapping[] = {
            {TreatFunctionsWithoutSpecification::None, "None"},
            {TreatFunctionsWithoutSpecification::OnlyUndefined,
             "OnlyUndefined"},
            {TreatFunctionsWithoutSpecification::All, "All"},
        };
    return {Mapping};
  }
};

namespace bugprone {
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
      TreatFunctionsWithoutSpecificationAsThrowing(
          Options.get("TreatFunctionsWithoutSpecificationAsThrowing",
                      TreatFunctionsWithoutSpecification::None)) {
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

  Tracer.assumeMissingDefinitionsFunctionsAsThrowing(
      TreatFunctionsWithoutSpecificationAsThrowing !=
      TreatFunctionsWithoutSpecification::None);

  Tracer.assumeUnannotatedFunctionsAsThrowing(
      TreatFunctionsWithoutSpecificationAsThrowing ==
      TreatFunctionsWithoutSpecification::All);
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
  Options.store(Opts, "TreatFunctionsWithoutSpecificationAsThrowing",
                TreatFunctionsWithoutSpecificationAsThrowing);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  auto MatchIf = [](bool Enabled, const auto &Matcher) {
    const ast_matchers::internal::Matcher<FunctionDecl> Nothing =
        unless(anything());
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

  if (Info.getBehaviour() != utils::ExceptionAnalyzer::State::Throwing)
    return;

  diag(MatchedDecl->getLocation(), "an exception may be thrown in function "
                                   "%0 which should not throw exceptions")
      << MatchedDecl;

  const utils::ExceptionAnalyzer::ExceptionInfo::Throwables &Exceptions =
      Info.getExceptions();
  const utils::ExceptionAnalyzer::ExceptionInfo::ThrowInfo *TI = nullptr;
  if (!Exceptions.empty())
    TI = &Exceptions.begin()->second;
  else if (Info.containsUnknownElements())
    TI = &Info.getUnknownThrowInfo();

  if (!TI || TI->Loc.isInvalid())
    return;

  if (!Exceptions.empty()) {
    const auto &[ThrowType, ThrowInfo] = *Exceptions.begin();
    diag(ThrowInfo.Loc,
         "frame #0: unhandled exception of type %0 may be thrown in function "
         "%1 here",
         DiagnosticIDs::Note)
        << QualType(ThrowType, 0U) << ThrowInfo.Stack.back().first;
  } else {
    diag(TI->Loc,
         "frame #0: an exception of unknown type may be thrown in function %0 "
         "here",
         DiagnosticIDs::Note)
        << TI->Stack.back().first;
  }

  size_t FrameNo = 1;
  for (auto CurrIt = ++TI->Stack.rbegin(), PrevIt = TI->Stack.rbegin();
       CurrIt != TI->Stack.rend(); ++CurrIt, ++PrevIt) {
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

} // namespace bugprone
} // namespace clang::tidy
