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

  // Support targeting
  //  - the switch statement itself (keyword, parens)
  //  - the whole expression (possibly wrapped in implicit casts)
  //  - the outer body (typically CompoundStmt)
  // Selections *within* the expression or body don't trigger.
  // direct child (the 
  Switch = CA->ASTNode.get<SwitchStmt>();
  if (!Switch) {
    if (const SelectionTree::Node *Parent = CA->outerImplicit().Parent)
      Switch = Parent->ASTNode.get<SwitchStmt>();
    if (!Switch)
      return false;
  }
  // Body need not be a CompoundStmt! But that's all we support editing.
  Body = llvm::dyn_cast_or_null<CompoundStmt>(Switch->getBody());
  if (!Body)
    return false;
  DeclCtx = &CA->getDeclContext();

  // Examine the condition of the switch statement to see if it's an enum.
  const Expr *Cond = Switch->getCond();
  if (!Cond)
    return false;
  // Ignore implicit casts, since enums implicitly cast to integer types.
  Cond = Cond->IgnoreParenImpCasts();
  // Get the canonical type to handle typedefs.
  EnumT = Cond->getType().getCanonicalType()->getAsAdjusted<EnumType>();
  if (!EnumT)
    return false;
  EnumD = EnumT->getDecl();
  if (!EnumD || EnumD->isDependentType())
    return false;

  // Finally, check which cases exist and which are covered.
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

    // Support for direct references to enum constants. This is required to
    // support C and ObjC which don't contain values in their ConstantExprs.
    // The general way to get the value of a case is EvaluateAsRValue, but we'd
    // rather not deal with that in case the AST is broken.
    if (auto *DRE = dyn_cast<DeclRefExpr>(CS->getLHS()->IgnoreParenCasts())) {
      if (auto *Enumerator = dyn_cast<EnumConstantDecl>(DRE->getDecl())) {
        auto Iter = ExpectedCases.find(Normalize(Enumerator->getInitVal()));
        if (Iter != ExpectedCases.end())
          Iter->second.setCovered();
        continue;
      }
    }

    // ConstantExprs with values are expected for C++, otherwise the storage
    // kind will be None.

    // Case expression is not a constant expression or is value-dependent,
    // so we may not be able to work out which cases are covered.
    const ConstantExpr *CE = dyn_cast<ConstantExpr>(CS->getLHS());
    if (!CE || CE->isValueDependent())
      return false;

    // We need a stored value in order to continue; currently both C and ObjC
    // enums won't have one.
    if (CE->getResultStorageKind() == ConstantResultStorageKind::None)
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
