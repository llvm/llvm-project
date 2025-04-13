//===--- ExceptionEscapeCheck.cpp - clang-tidy ----------------------------===//
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
      RawIgnoredExceptions(Options.get("IgnoredExceptions", "")) {
  llvm::SmallVector<StringRef, 8> FunctionsThatShouldNotThrowVec,
      IgnoredExceptionsVec;
  StringRef(RawFunctionsThatShouldNotThrow)
      .split(FunctionsThatShouldNotThrowVec, ",", -1, false);
  FunctionsThatShouldNotThrow.insert_range(FunctionsThatShouldNotThrowVec);

  llvm::StringSet<> IgnoredExceptions;
  StringRef(RawIgnoredExceptions).split(IgnoredExceptionsVec, ",", -1, false);
  IgnoredExceptions.insert_range(IgnoredExceptionsVec);
  Tracer.ignoreExceptions(std::move(IgnoredExceptions));
  Tracer.ignoreBadAlloc(true);
}

void ExceptionEscapeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionsThatShouldNotThrow",
                RawFunctionsThatShouldNotThrow);
  Options.store(Opts, "IgnoredExceptions", RawIgnoredExceptions);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(
          isDefinition(),
          anyOf(isNoThrow(),
                allOf(anyOf(cxxDestructorDecl(),
                            cxxConstructorDecl(isMoveConstructor()),
                            cxxMethodDecl(isMoveAssignmentOperator()), isMain(),
                            allOf(hasAnyName("swap", "iter_swap", "iter_move"),
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

  if (Info.getBehaviour() != utils::ExceptionAnalyzer::State::Throwing) {
    return;
  }

  diag(MatchedDecl->getLocation(), "an exception may be thrown in function "
                                   "%0 which should not throw exceptions")
      << MatchedDecl;

  const utils::ExceptionAnalyzer::ExceptionInfo::ThrowInfo ThrowInfo =
      Info.getExceptions().begin()->getSecond();

  if (ThrowInfo.Loc.isInvalid()) {
    return;
  }

  // FIXME: We should provide exact position of functions calls, not only call
  // stack of thrown exception.
  const utils::ExceptionAnalyzer::CallStack &Stack = ThrowInfo.Stack;
  diag(Stack.front()->getLocation(),
       "throw stack of unhandled exception, starting from function %0",
       DiagnosticIDs::Note)
      << Stack.front();

  size_t FrameNo = 0;
  for (const FunctionDecl *CallNode : Stack) {
    if (FrameNo != Stack.size() - 1) {
      diag(CallNode->getLocation(), "frame #%0: function %1",
           DiagnosticIDs::Note)
          << FrameNo << CallNode;
    } else {
      diag(ThrowInfo.Loc,
           "frame #%0: function %1 throws unhandled exception here",
           DiagnosticIDs::Note)
          << FrameNo << CallNode;
    }
    ++FrameNo;
  }
}

} // namespace clang::tidy::bugprone
