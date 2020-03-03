//===--- ExceptionEscapeCheck.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ExceptionEscapeCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace openmp {

ExceptionEscapeCheck::ExceptionEscapeCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RawIgnoredExceptions(Options.get("IgnoredExceptions", "")) {
  llvm::SmallVector<StringRef, 8> FunctionsThatShouldNotThrowVec,
      IgnoredExceptionsVec;

  llvm::StringSet<> IgnoredExceptions;
  StringRef(RawIgnoredExceptions).split(IgnoredExceptionsVec, ",", -1, false);
  llvm::transform(IgnoredExceptionsVec, IgnoredExceptionsVec.begin(),
                  [](StringRef S) { return S.trim(); });
  IgnoredExceptions.insert(IgnoredExceptionsVec.begin(),
                           IgnoredExceptionsVec.end());
  Tracer.ignoreExceptions(std::move(IgnoredExceptions));
  Tracer.ignoreBadAlloc(true);
}

void ExceptionEscapeCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IgnoredExceptions", RawIgnoredExceptions);
}

void ExceptionEscapeCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(ompExecutableDirective(
                         unless(isStandaloneDirective()),
                         hasStructuredBlock(stmt().bind("structured-block")))
                         .bind("directive"),
                     this);
}

void ExceptionEscapeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Directive =
      Result.Nodes.getNodeAs<OMPExecutableDirective>("directive");
  assert(Directive && "Expected to match some OpenMP Executable directive.");
  const auto *StructuredBlock =
      Result.Nodes.getNodeAs<Stmt>("structured-block");
  assert(StructuredBlock && "Expected to get some OpenMP Structured Block.");

  if (Tracer.analyze(StructuredBlock).getBehaviour() !=
      utils::ExceptionAnalyzer::State::Throwing)
    return; // No exceptions have been proven to escape out of the struc. block.

  // FIXME: We should provide more information about the exact location where
  // the exception is thrown, maybe the full path the exception escapes.

  diag(StructuredBlock->getBeginLoc(),
       "an exception thrown inside of the OpenMP '%0' region is not caught in "
       "that same region")
      << getOpenMPDirectiveName(Directive->getDirectiveKind());
}

} // namespace openmp
} // namespace tidy
} // namespace clang
