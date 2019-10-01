//===--- ExceptionEscapeCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionEscapeCheck.h"

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"

using namespace clang::ast_matchers;

namespace clang {
namespace {
AST_MATCHER_P(FunctionDecl, isEnabled, llvm::StringSet<>,
              FunctionsThatShouldNotThrow) {
  return FunctionsThatShouldNotThrow.count(Node.getNameAsString()) > 0;
}
} // namespace

namespace tidy {
namespace bugprone {
ExceptionEscapeCheck::ExceptionEscapeCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context), RawFunctionsThatShouldNotThrow(Options.get(
                                         "FunctionsThatShouldNotThrow", "")),
      RawIgnoredExceptions(Options.get("IgnoredExceptions", "")) {
  llvm::SmallVector<StringRef, 8> FunctionsThatShouldNotThrowVec,
      IgnoredExceptionsVec;
  StringRef(RawFunctionsThatShouldNotThrow)
      .split(FunctionsThatShouldNotThrowVec, ",", -1, false);
  FunctionsThatShouldNotThrow.insert(FunctionsThatShouldNotThrowVec.begin(),
                                     FunctionsThatShouldNotThrowVec.end());

  llvm::StringSet<> IgnoredExceptions;
  StringRef(RawIgnoredExceptions).split(IgnoredExceptionsVec, ",", -1, false);
  IgnoredExceptions.insert(IgnoredExceptionsVec.begin(),
                           IgnoredExceptionsVec.end());
  Tracer.ignoreExceptions(std::move(IgnoredExceptions));
  Tracer.ignoreBadAlloc(true);
}

void ExceptionEscapeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "FunctionsThatShouldNotThrow",
                RawFunctionsThatShouldNotThrow);
  Options.store(Opts, "IgnoredExceptions", RawIgnoredExceptions);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  if (!getLangOpts().CPlusPlus || !getLangOpts().CXXExceptions)
    return;

  Finder->addMatcher(
      functionDecl(anyOf(isNoThrow(), cxxDestructorDecl(),
                         cxxConstructorDecl(isMoveConstructor()),
                         cxxMethodDecl(isMoveAssignmentOperator()),
                         hasName("main"), hasName("swap"),
                         isEnabled(FunctionsThatShouldNotThrow)))
          .bind("thrower"),
      this);
}

void ExceptionEscapeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<FunctionDecl>("thrower");

  if (!MatchedDecl)
    return;

  if (Tracer.analyze(MatchedDecl).getBehaviour() ==
      utils::ExceptionAnalyzer::State::Throwing)
    // FIXME: We should provide more information about the exact location where
    // the exception is thrown, maybe the full path the exception escapes
    diag(MatchedDecl->getLocation(),
         "an exception may be thrown in function %0 "

         "which should not throw exceptions")
        << MatchedDecl;
}

} // namespace bugprone
} // namespace tidy
} // namespace clang
