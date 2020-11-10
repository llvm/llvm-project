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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
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
    return CodeAction::QUICKFIX_KIND;
  }

private:
  class ExpectedCase {
  public:
    ExpectedCase(const EnumConstantDecl *Decl) : Data(Decl, false) {}
    bool isCovered() const { return Data.getInt(); }
    void setCovered(bool Val = true) { Data.setInt(Val); }
    const EnumConstantDecl *getEnumConstant() const {
      return Data.getPointer();
    }

  private:
    llvm::PointerIntPair<const EnumConstantDecl *, 1, bool> Data;
  };

  const DeclContext *DeclCtx = nullptr;
  const SwitchStmt *Switch = nullptr;
  const CompoundStmt *Body = nullptr;
  const EnumType *EnumT = nullptr;
  const EnumDecl *EnumD = nullptr;
  // Maps the Enum values to the EnumConstantDecl and a bool signifying if its
  // covered in the switch.
  llvm::MapVector<llvm::APSInt, ExpectedCase> ExpectedCases;
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

  // We trigger if there are any values in the enum that aren't covered by the
  // switch.

  ASTContext &Ctx = Sel.AST->getASTContext();

  unsigned EnumIntWidth = Ctx.getIntWidth(QualType(EnumT, 0));
  bool EnumIsSigned = EnumT->isSignedIntegerOrEnumerationType();

  auto Normalize = [&](llvm::APSInt Val) {
    Val = Val.extOrTrunc(EnumIntWidth);
    Val.setIsSigned(EnumIsSigned);
    return Val;
  };

  for (auto *EnumConstant : EnumD->enumerators()) {
    ExpectedCases.insert(
        std::make_pair(Normalize(EnumConstant->getInitVal()), EnumConstant));
  }

  for (const SwitchCase *CaseList = Switch->getSwitchCaseList(); CaseList;
       CaseList = CaseList->getNextSwitchCase()) {
    // Default likely intends to cover cases we'd insert.
    if (isa<DefaultStmt>(CaseList))
      return false;

    const CaseStmt *CS = cast<CaseStmt>(CaseList);

    // GNU range cases are rare, we don't support them.
    if (CS->caseStmtIsGNURange())
      return false;

    // Case expression is not a constant expression or is value-dependent,
    // so we may not be able to work out which cases are covered.
    const ConstantExpr *CE = dyn_cast<ConstantExpr>(CS->getLHS());
    if (!CE || CE->isValueDependent())
      return false;

    // Unsure if this case could ever come up, but prevents an unreachable
    // executing in getResultAsAPSInt.
    if (CE->getResultStorageKind() == ConstantExpr::RSK_None)
      return false;
    auto Iter = ExpectedCases.find(Normalize(CE->getResultAsAPSInt()));
    if (Iter != ExpectedCases.end())
      Iter->second.setCovered();
  }

  return !llvm::all_of(ExpectedCases,
                       [](auto &Pair) { return Pair.second.isCovered(); });
}

Expected<Tweak::Effect> PopulateSwitch::apply(const Selection &Sel) {
  ASTContext &Ctx = Sel.AST->getASTContext();

  SourceLocation Loc = Body->getRBracLoc();
  ASTContext &DeclASTCtx = DeclCtx->getParentASTContext();

  llvm::SmallString<256> Text;
  for (auto &EnumConstant : ExpectedCases) {
    // Skip any enum constants already covered
    if (EnumConstant.second.isCovered())
      continue;

    Text.append({"case ", getQualification(DeclASTCtx, DeclCtx, Loc, EnumD)});
    if (EnumD->isScoped())
      Text.append({EnumD->getName(), "::"});
    Text.append({EnumConstant.second.getEnumConstant()->getName(), ":"});
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
