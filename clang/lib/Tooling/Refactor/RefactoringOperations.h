//===--- RefactoringOperations.h - The supported refactoring operations ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORINGOPERATIONS_H
#define LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORINGOPERATIONS_H

#include "ASTSlice.h"
#include "clang/Tooling/Refactor/RefactoringOperation.h"

namespace clang {

class Expr;
class IfStmt;
class VarDecl;

namespace tooling {

#define REFACTORING_OPERATION_ACTION(Name, Spelling, Command)                  \
  RefactoringOperationResult initiate##Name##Operation(                        \
      ASTSlice &Slice, ASTContext &Context, SourceLocation Location,           \
      SourceRange SelectionRange, bool CreateOperation = true);
#define REFACTORING_OPERATION_SUB_ACTION(Name, Parent, Spelling, Command)      \
  RefactoringOperationResult initiate##Parent##Name##Operation(                \
      ASTSlice &Slice, ASTContext &Context, SourceLocation Location,           \
      SourceRange SelectionRange, bool CreateOperation = true);
#include "clang/Tooling/Refactor/RefactoringActions.def"

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_LIB_TOOLING_REFACTOR_REFACTORINGOPERATIONS_H
