//===--- NoSuspendWithLockCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NoSuspendWithLockCheck.h"
#include "../utils/ExprSequence.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Analysis/CFG.h"

using namespace clang::ast_matchers;

namespace clang::tidy::cppcoreguidelines {

void NoSuspendWithLockCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "LockGuards", LockGuards);
}

void NoSuspendWithLockCheck::registerMatchers(MatchFinder *Finder) {
  auto LockType = templateSpecializationType(
      hasDeclaration(namedDecl(matchers::matchesAnyListedName(
          utils::options::parseStringList(LockGuards)))));

  StatementMatcher Lock =
      declStmt(has(varDecl(hasType(LockType)).bind("lock-decl")))
          .bind("lock-decl-stmt");
  Finder->addMatcher(
      expr(anyOf(coawaitExpr(), coyieldExpr(), dependentCoawaitExpr()),
           forCallable(functionDecl().bind("function")),
           unless(isInTemplateInstantiation()),
           hasAncestor(
               compoundStmt(has(Lock), forCallable(equalsBoundNode("function")))
                   .bind("block")))
          .bind("suspend"),
      this);
}

void NoSuspendWithLockCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Block = Result.Nodes.getNodeAs<CompoundStmt>("block");
  const auto *Suspend = Result.Nodes.getNodeAs<Expr>("suspend");
  const auto *LockDecl = Result.Nodes.getNodeAs<VarDecl>("lock-decl");
  const auto *LockStmt = Result.Nodes.getNodeAs<Stmt>("lock-decl-stmt");

  if (!Block || !Suspend || !LockDecl || !LockStmt)
    return;

  ASTContext &Context = *Result.Context;
  CFG::BuildOptions Options;
  Options.AddImplicitDtors = true;
  Options.AddTemporaryDtors = true;

  std::unique_ptr<CFG> TheCFG = CFG::buildCFG(
      nullptr, const_cast<clang::CompoundStmt *>(Block), &Context, Options);
  if (!TheCFG)
    return;

  utils::ExprSequence Sequence(TheCFG.get(), Block, &Context);
  const Stmt *LastBlockStmt = Block->body_back();
  if (Sequence.inSequence(LockStmt, Suspend) &&
      (Suspend == LastBlockStmt ||
       Sequence.inSequence(Suspend, LastBlockStmt))) {
    diag(Suspend->getBeginLoc(), "coroutine suspended with lock %0 held")
        << LockDecl;
  }
}

} // namespace clang::tidy::cppcoreguidelines
