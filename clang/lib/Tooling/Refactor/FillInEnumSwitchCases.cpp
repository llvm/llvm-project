//===--- FillInEnumSwitchCases.cpp -  -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the "Add missing switch cases" refactoring operation.
//
//===----------------------------------------------------------------------===//

#include "RefactoringOperations.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/Edit/RefactoringFixits.h"

using namespace clang;
using namespace clang::tooling;

namespace {

class FillInEnumSwitchCasesOperation : public RefactoringOperation {
public:
  FillInEnumSwitchCasesOperation(const EnumDecl *Enum, const SwitchStmt *Switch,
                                 const DeclContext *SwitchContext)
      : Enum(Enum), Switch(Switch), SwitchContext(SwitchContext) {}

  const Stmt *getTransformedStmt() const override { return Switch; }

  llvm::Expected<RefactoringResult>
  perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex) override;

  const EnumDecl *Enum;
  const SwitchStmt *Switch;
  const DeclContext *SwitchContext;
};

} // end anonymous namespace

RefactoringOperationResult
clang::tooling::initiateFillInEnumSwitchCasesOperation(
    ASTSlice &Slice, ASTContext &Context, SourceLocation Location,
    SourceRange SelectionRange, bool CreateOperation) {
  const SwitchStmt *Switch;
  const Decl *ParentDecl;
  if (SelectionRange.isValid()) {
    auto SelectedSet = Slice.getSelectedStmtSet();
    if (!SelectedSet)
      return None;
    Switch = dyn_cast_or_null<SwitchStmt>(SelectedSet->containsSelectionRange);
    // FIXME: Improve the interface for this to make it similar to SelectedStmt
    if (SelectedSet->containsSelectionRange)
      ParentDecl =
          Slice.parentDeclForIndex(*SelectedSet->containsSelectionRangeIndex);
  } else {
    auto SelectedStmt = Slice.nearestSelectedStmt(Stmt::SwitchStmtClass);
    if (!SelectedStmt)
      return None;
    Switch = cast<SwitchStmt>(SelectedStmt->getStmt());
    ParentDecl = SelectedStmt->getParentDecl();
  }
  if (!Switch)
    return None;

  // Ensure that the type is an enum.
  const Expr *Cond = Switch->getCond()->IgnoreImpCasts();
  const EnumDecl *ED = nullptr;
  if (const auto *ET = Cond->getType()->getAs<EnumType>())
    ED = ET->getDecl();
  else {
    // Enum literals are 'int' in C.
    if (const auto *DRE = dyn_cast<DeclRefExpr>(Cond)) {
      if (const auto *EC = dyn_cast<EnumConstantDecl>(DRE->getDecl()))
        ED = dyn_cast<EnumDecl>(EC->getDeclContext());
    }
  }

  if (!ED)
    return RefactoringOperationResult("The switch doesn't operate on an enum");
  if (!ED->isCompleteDefinition())
    return RefactoringOperationResult("The enum type is incomplete");

  if (Switch->isAllEnumCasesCovered())
    return RefactoringOperationResult("All enum cases are already covered");

  RefactoringOperationResult Result;
  Result.Initiated = true;
  if (!CreateOperation)
    return Result;
  auto Operation = llvm::make_unique<FillInEnumSwitchCasesOperation>(
      ED, Switch, dyn_cast<DeclContext>(ParentDecl));
  Result.RefactoringOp = std::move(Operation);
  return Result;
}

llvm::Expected<RefactoringResult>
FillInEnumSwitchCasesOperation::perform(ASTContext &Context,
                                        const Preprocessor &ThePreprocessor,
                                        const RefactoringOptionSet &Options,
                                        unsigned SelectedCandidateIndex) {
  std::vector<RefactoringReplacement> Replacements;
  edit::fillInMissingSwitchEnumCases(
      Context, Switch, Enum, SwitchContext,
      [&](const FixItHint &Hint) { Replacements.push_back(Hint); });
  return std::move(Replacements);
}
