//===--- PopulateSwitch.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tweak that populates an empty switch statement of an enumeration type with
// all of the enumerators of that type.
//
// Before:
//   enum Color { RED, GREEN, BLUE };
//
//   void f(Color color) {
//     switch (color) {}
//   }
//
// After:
//   enum Color { RED, GREEN, BLUE };
//
//   void f(Color color) {
//     switch (color) {
//     case RED:
//     case GREEN:
//     case BLUE:
//       break;
//     }
//   }
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Selection.h"
#include "refactor/Tweak.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include <string>

namespace clang {
namespace clangd {
namespace {
class PopulateSwitch : public Tweak {
  const char *id() const override;
  bool prepare(const Selection &Sel) override;
  Expected<Effect> apply(const Selection &Sel) override;
  std::string title() const override { return "Populate switch"; }
  Intent intent() const override { return Refactor; }

private:
  ASTContext *ASTCtx = nullptr;
  const DeclContext *DeclCtx = nullptr;
  const SwitchStmt *Switch = nullptr;
  const CompoundStmt *Body = nullptr;
  const EnumDecl *EnumD = nullptr;
};

REGISTER_TWEAK(PopulateSwitch)

bool PopulateSwitch::prepare(const Selection &Sel) {
  ASTCtx = &Sel.AST->getASTContext();

  const SelectionTree::Node *CA = Sel.ASTSelection.commonAncestor();
  if (!CA)
    return false;

  const Stmt *CAStmt = CA->ASTNode.get<Stmt>();
  if (!CAStmt)
    return false;

  // Go up a level if we see a compound statement.
  // switch (value) {}
  //                ^^
  if (isa<CompoundStmt>(CAStmt)) {
    CA = CA->Parent;
    if (!CA)
      return false;

    CAStmt = CA->ASTNode.get<Stmt>();
    if (!CAStmt)
      return false;
  }

  DeclCtx = &CA->getDeclContext();
  Switch = dyn_cast<SwitchStmt>(CAStmt);
  if (!Switch)
    return false;

  Body = dyn_cast<CompoundStmt>(Switch->getBody());
  if (!Body)
    return false;

  // Since we currently always insert all enumerators, don't suggest this tweak
  // if the body is not empty.
  if (!Body->body_empty())
    return false;

  const Expr *Cond = Switch->getCond();
  if (!Cond)
    return false;

  // Ignore implicit casts, since enums implicitly cast to integer types.
  Cond = Cond->IgnoreParenImpCasts();

  const EnumType *EnumT = Cond->getType()->getAsAdjusted<EnumType>();
  if (!EnumT)
    return false;

  EnumD = EnumT->getDecl();
  if (!EnumD)
    return false;

  // If there aren't any enumerators, there's nothing to insert.
  if (EnumD->enumerator_begin() == EnumD->enumerator_end())
    return false;

  return true;
}

Expected<Tweak::Effect> PopulateSwitch::apply(const Selection &Sel) {
  const SourceManager &SM = ASTCtx->getSourceManager();
  SourceLocation Loc = Body->getRBracLoc();

  std::string Text;
  for (EnumConstantDecl *Enumerator : EnumD->enumerators()) {
    Text += "case ";
    Text += getQualification(*ASTCtx, DeclCtx, Loc, EnumD);
    if (EnumD->isScoped()) {
      Text += EnumD->getName();
      Text += "::";
    }
    Text += Enumerator->getName();
    Text += ":";
  }
  Text += "break;";

  return Effect::mainFileEdit(
      SM, tooling::Replacements(tooling::Replacement(SM, Loc, 0, Text)));
}
} // namespace
} // namespace clangd
} // namespace clang
