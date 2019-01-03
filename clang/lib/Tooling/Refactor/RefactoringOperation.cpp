//===--- RefactoringOperation.cpp - Defines a refactoring operation -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Refactor/RefactoringOperation.h"
#include "ASTSlice.h"
#include "RefactoringOperations.h"
#include "SourceLocationUtilities.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTContext.h"
#include "clang/Tooling/Refactor/SymbolOperation.h"
#include "clang/Tooling/Refactor/USRFinder.h"
#include "llvm/Support/Errc.h"

using namespace clang;
using namespace clang::tooling;

char RefactoringOperationError::ID;

void RefactoringOperationError::log(raw_ostream &OS) const {
  OS << "Refactoring operation failed: " << FailureReason;
}

std::error_code RefactoringOperationError::convertToErrorCode() const {
  return make_error_code(llvm::errc::operation_not_permitted);
}

RefactoringOperationResult clang::tooling::initiateRefactoringOperationAt(
    SourceLocation Location, SourceRange SelectionRange, ASTContext &Context,
    RefactoringActionType ActionType, bool CreateOperation) {
  if (Location.isInvalid())
    return None;
  if (ActionType == RefactoringActionType::Rename ||
      ActionType == RefactoringActionType::Rename_Local) {
    const NamedDecl *FoundDecl = rename::getNamedDeclAt(Context, Location);
    if (!FoundDecl)
      return None;
    RefactoringOperationResult Result;
    Result.Initiated = true;
    if (CreateOperation)
      Result.SymbolOp = llvm::make_unique<SymbolOperation>(FoundDecl, Context);
    return Result;
  }
  SourceManager &SM = Context.getSourceManager();
  if (Location.isMacroID())
    Location = SM.getSpellingLoc(Location);
  assert(Location.isFileID() && "Invalid location");

  // TODO: Don't perform duplicate work when initiateRefactoringOperationAt is
  // called from findRefactoringActionsAt.
  if (SelectionRange.isValid()) {
    if (SelectionRange.getBegin().isMacroID() ||
        SelectionRange.getEnd().isMacroID())
      SelectionRange = SourceRange(SM.getSpellingLoc(SelectionRange.getBegin()),
                                   SM.getSpellingLoc(SelectionRange.getEnd()));
    SelectionRange = trimSelectionRange(
        SelectionRange, Context.getSourceManager(), Context.getLangOpts());
  }
  ASTSlice Slice(Location, SelectionRange, Context);

  switch (ActionType) {
#define REFACTORING_OPERATION_ACTION(Name, Spelling, Command)                  \
  case RefactoringActionType::Name:                                            \
    return initiate##Name##Operation(Slice, Context, Location, SelectionRange, \
                                     CreateOperation);
#define REFACTORING_OPERATION_SUB_ACTION(Name, Parent, Spelling, Command)      \
  case RefactoringActionType::Parent##_##Name:                                 \
    return initiate##Parent##Name##Operation(Slice, Context, Location,         \
                                             SelectionRange, CreateOperation);
#include "clang/Tooling/Refactor/RefactoringActions.def"
  default:
    break;
  }
  return RefactoringOperationResult();
}

RefactoringOperationResult clang::tooling::initiateRefactoringOperationOnDecl(
    StringRef DeclUSR, ASTContext &Context, RefactoringActionType ActionType) {
  if (ActionType != RefactoringActionType::Rename)
    return None;
  const NamedDecl *FoundDecl = rename::getNamedDeclWithUSR(Context, DeclUSR);
  if (!FoundDecl)
    return None;
  RefactoringOperationResult Result;
  Result.Initiated = true;
  Result.SymbolOp = llvm::make_unique<SymbolOperation>(FoundDecl, Context);
  return Result;
}
