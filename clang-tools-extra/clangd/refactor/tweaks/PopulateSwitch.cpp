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
#include "support/Logger.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/ADT/SmallSet.h"
#include <cassert>
#include <string>

namespace clang {
namespace clangd {
namespace {
class PopulateSwitch : public Tweak {
  const char *id() const override;
  bool prepare(const Selection &Sel) override;
  Expected<Effect> apply(const Selection &Sel) override;
  std::string title() const override { return "Populate switch"; }
  llvm::StringLiteral kind() const override {
    return CodeAction::REFACTOR_KIND;
  }

private:
  const DeclContext *DeclCtx = nullptr;
  const SwitchStmt *Switch = nullptr;
  const CompoundStmt *Body = nullptr;
  const EnumType *EnumT = nullptr;
  const EnumDecl *EnumD = nullptr;
};

REGISTER_TWEAK(PopulateSwitch)

bool PopulateSwitch::prepare(const Selection &Sel) {
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

  const Expr *Cond = Switch->getCond();
  if (!Cond)
    return false;

  // Ignore implicit casts, since enums implicitly cast to integer types.
  Cond = Cond->IgnoreParenImpCasts();

  EnumT = Cond->getType()->getAsAdjusted<EnumType>();
  if (!EnumT)
    return false;

  EnumD = EnumT->getDecl();
  if (!EnumD)
    return false;

  // We trigger if there are fewer cases than enum values (and no case covers
  // multiple values). This guarantees we'll have at least one case to insert.
  // We don't yet determine what the cases are, as that means evaluating
  // expressions.
  auto I = EnumD->enumerator_begin();
  auto E = EnumD->enumerator_end();

  for (const SwitchCase *CaseList = Switch->getSwitchCaseList();
       CaseList && I != E; CaseList = CaseList->getNextSwitchCase(), I++) {
    // Default likely intends to cover cases we'd insert.
    if (isa<DefaultStmt>(CaseList))
      return false;

    const CaseStmt *CS = cast<CaseStmt>(CaseList);
    // Case statement covers multiple values, so just counting doesn't work.
    if (CS->caseStmtIsGNURange())
      return false;

    // Case expression is not a constant expression or is value-dependent,
    // so we may not be able to work out which cases are covered.
    const ConstantExpr *CE = dyn_cast<ConstantExpr>(CS->getLHS());
    if (!CE || CE->isValueDependent())
      return false;
  }

  // Only suggest tweak if we have more enumerators than cases.
  return I != E;
}

Expected<Tweak::Effect> PopulateSwitch::apply(const Selection &Sel) {
  ASTContext &Ctx = Sel.AST->getASTContext();

  // Get the enum's integer width and signedness, for adjusting case literals.
  unsigned EnumIntWidth = Ctx.getIntWidth(QualType(EnumT, 0));
  bool EnumIsSigned = EnumT->isSignedIntegerOrEnumerationType();

  llvm::SmallSet<llvm::APSInt, 32> ExistingEnumerators;
  for (const SwitchCase *CaseList = Switch->getSwitchCaseList(); CaseList;
       CaseList = CaseList->getNextSwitchCase()) {
    const CaseStmt *CS = cast<CaseStmt>(CaseList);
    assert(!CS->caseStmtIsGNURange());
    const ConstantExpr *CE = cast<ConstantExpr>(CS->getLHS());
    assert(!CE->isValueDependent());
    llvm::APSInt Val = CE->getResultAsAPSInt();
    Val = Val.extOrTrunc(EnumIntWidth);
    Val.setIsSigned(EnumIsSigned);
    ExistingEnumerators.insert(Val);
  }

  SourceLocation Loc = Body->getRBracLoc();
  ASTContext &DeclASTCtx = DeclCtx->getParentASTContext();

  std::string Text;
  for (EnumConstantDecl *Enumerator : EnumD->enumerators()) {
    if (ExistingEnumerators.contains(Enumerator->getInitVal()))
      continue;

    Text += "case ";
    Text += getQualification(DeclASTCtx, DeclCtx, Loc, EnumD);
    if (EnumD->isScoped()) {
      Text += EnumD->getName();
      Text += "::";
    }
    Text += Enumerator->getName();
    Text += ":";
  }

  assert(!Text.empty() && "No enumerators to insert!");
  Text += "break;";

  const SourceManager &SM = Ctx.getSourceManager();
  return Effect::mainFileEdit(
      SM, tooling::Replacements(tooling::Replacement(SM, Loc, 0, Text)));
}
} // namespace
} // namespace clangd
} // namespace clang
