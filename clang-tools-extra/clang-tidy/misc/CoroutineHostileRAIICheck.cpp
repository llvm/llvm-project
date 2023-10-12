//===--- CoroutineHostileRAII.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoroutineHostileRAIICheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::misc {
CoroutineHostileRAIICheck::CoroutineHostileRAIICheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {
  for (StringRef Denied :
       utils::options::parseStringList(Options.get("RAIIDenyList", ""))) {
    Denied.consume_front("::");
    RAIIDenyList.push_back(Denied);
  }
}

void CoroutineHostileRAIICheck::registerMatchers(MatchFinder *Finder) {
  // A suspension happens with co_await or co_yield.
  Finder->addMatcher(coawaitExpr().bind("suspension"), this);
  Finder->addMatcher(coyieldExpr().bind("suspension"), this);
}

void CoroutineHostileRAIICheck::checkVarDecl(const VarDecl *VD) {
  RecordDecl *RD = VD->getType().getCanonicalType()->getAsRecordDecl();
  if (RD->hasAttr<clang::ScopedLockableAttr>()) {
    diag(VD->getLocation(),
         "%0 holds a lock across a suspension point of coroutine and could be "
         "unlocked by a different thread")
        << VD;
  }
  if (std::find(RAIIDenyList.begin(), RAIIDenyList.end(),
                RD->getQualifiedNameAsString()) != RAIIDenyList.end()) {
    diag(VD->getLocation(),
         "%0 persists across a suspension point of coroutine")
        << VD;
  }
}

void CoroutineHostileRAIICheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Suspension = Result.Nodes.getNodeAs<Stmt>("suspension");
  DynTypedNode P;
  for (const Stmt *Child = Suspension; Child; Child = P.get<Stmt>()) {
    auto Parents = Result.Context->getParents(*Child);
    if (Parents.empty())
      break;
    P = *Parents.begin();
    auto *PCS = P.get<CompoundStmt>();
    if (!PCS)
      continue;
    for (auto Sibling = PCS->child_begin();
         *Sibling != Child && Sibling != PCS->child_end(); ++Sibling) {
      if (auto *DS = dyn_cast<DeclStmt>(*Sibling)) {
        for (Decl *D : DS->decls()) {
          if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
            checkVarDecl(VD);
          }
        }
      }
    }
  }
}

void CoroutineHostileRAIICheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "RAIIDenyList",
                utils::options::serializeStringList(RAIIDenyList));
}
} // namespace clang::tidy::misc
