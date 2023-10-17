//===--- CoroutineHostileRAII.cpp - clang-tidy ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoroutineHostileRAIICheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/DiagnosticIDs.h"

using namespace clang::ast_matchers;
namespace clang::tidy::misc {
using ::clang::ast_matchers::internal::ASTMatchFinder;
using clang::ast_matchers::internal::BoundNodesTreeBuilder;

// In case of a match, add the bindings as a separate match. Also don't clear
// the bindings if a match is not found (unlike Matcher::matches).
template <class Matcher, class Node>
bool match(Matcher M, Node &N, ASTMatchFinder *Finder,
           BoundNodesTreeBuilder *Builder) {
  BoundNodesTreeBuilder LocalBuilder;
  if (M.matches(N, Finder, &LocalBuilder)) {
    Builder->addMatch(LocalBuilder);
    return true;
  }
  return false;
}

CoroutineHostileRAIICheck::CoroutineHostileRAIICheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      RAIITypesList(utils::options::parseStringList(
          Options.get("RAIITypesList", "std::lock_guard;std::scoped_lock"))) {}

AST_MATCHER_P(VarDecl, isHostileRAII, std::vector<StringRef>, RAIITypesList) {
  return match(varDecl(hasType(hasCanonicalType(hasDeclaration(
                           hasAttr(attr::Kind::ScopedLockable)))))
                   .bind("scoped-lockable"),
               Node, Finder, Builder) ||
         match(varDecl(hasType(hasCanonicalType(hasDeclaration(
                           namedDecl(hasAnyName(RAIITypesList))))))
                   .bind("raii"),
               Node, Finder, Builder);
}

AST_MATCHER_P(DeclStmt, hasHostileRAII, std::vector<StringRef>, RAIITypesList) {
  bool IsHostile = false;
  for (Decl *D : Node.decls())
    IsHostile |=
        match(varDecl(isHostileRAII(RAIITypesList)), *D, Finder, Builder);
  return IsHostile;
}

AST_MATCHER_P(Expr, isHostileSuspension, std::vector<StringRef>,
              RAIITypesList) {
  DynTypedNode P;
  bool IsHostile = false;
  for (const Stmt *Child = &Node; Child; Child = P.get<Stmt>()) {
    auto Parents = Finder->getASTContext().getParents(*Child);
    if (Parents.empty())
      break;
    P = *Parents.begin();
    auto *PCS = P.get<CompoundStmt>();
    if (!PCS)
      continue;
    for (const auto &Sibling : PCS->children()) {
      // Child contains suspension. Siblings after Child do not persist across
      // this suspension.
      if (Sibling == Child)
        break;
      IsHostile |= match(declStmt(hasHostileRAII(RAIITypesList)), *Sibling,
                         Finder, Builder);
    }
  }
  return IsHostile;
}

void CoroutineHostileRAIICheck::registerMatchers(MatchFinder *Finder) {
  // A suspension happens with co_await or co_yield.
  Finder->addMatcher(
      coawaitExpr(isHostileSuspension(RAIITypesList)).bind("suspension"), this);
  Finder->addMatcher(
      coyieldExpr(isHostileSuspension(RAIITypesList)).bind("suspension"), this);
}

void CoroutineHostileRAIICheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("scoped-lockable"))
    diag(VD->getLocation(),
         "%0 holds a lock across a suspension point of coroutine and could be "
         "unlocked by a different thread")
        << VD;
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("raii"))
    diag(VD->getLocation(),
         "%0 persists across a suspension point of coroutine")
        << VD;
  if (const auto *Suspension = Result.Nodes.getNodeAs<Expr>("suspension"))
    diag(Suspension->getBeginLoc(), "suspension point is here",
         DiagnosticIDs::Note);
}

void CoroutineHostileRAIICheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "RAIITypesList",
                utils::options::serializeStringList(RAIITypesList));
}
} // namespace clang::tidy::misc
