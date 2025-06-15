//===--- ExecutionVisitor.h - clang-tidy ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXECUTIONVISITOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_EXECUTIONVISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace clang::tidy::utils {

/// Helper class that can be used to traverse all statements (including
/// expressions) that can execute while executing a given statement.
template <typename T> class ExecutionVisitor : public RecursiveASTVisitor<T> {
public:
  ExecutionVisitor() : IsInFunction(false) {}

protected:
  void traverseExecution(Stmt *S) {
    FunctionsToBeChecked.clear();
    IsInFunction = false;
    RecursiveASTVisitor<T>::TraverseStmt(S);

    // We keep a list of functions to be checked during traversal so that they
    // are not checked multiple times. If this weren't the case, we would get
    // infinite recursion on recursive functions.
    traverseFunctionsToBeChecked();
  }

  bool isInFunction() const { return IsInFunction; }

  void checkFunctionLater(const FunctionDecl *FD) {
    if (!FD->hasBody())
      return;

    for (const FunctionDecl *Fun : FunctionsToBeChecked)
      if (Fun->getDeclName() == FD->getDeclName())
        return;

    FunctionsToBeChecked.push_back(FD);
  }

  void checkDestructorLater(const CXXRecordDecl *D) {
    if (!D->hasDefinition() || D->hasIrrelevantDestructor())
      return;

    const CXXMethodDecl *Destructor = D->getDestructor();
    checkFunctionLater(static_cast<const FunctionDecl *>(Destructor));

    // We recurse into struct/class members and base classes, as their
    // destructors will run as well.

    for (const FieldDecl *F : D->fields()) {
      const Type *FieldType = F->getType().getTypePtrOrNull();
      if (!FieldType) {
        continue;
      }

      const CXXRecordDecl *FieldRecordDecl = FieldType->getAsCXXRecordDecl();
      if (!FieldRecordDecl)
        continue;

      checkDestructorLater(FieldRecordDecl);
    }

    for (const CXXBaseSpecifier Base : D->bases()) {
      const Type *BaseType = Base.getType().getTypePtrOrNull();
      if (!BaseType)
        continue;

      const CXXRecordDecl *BaseRecordDecl = BaseType->getAsCXXRecordDecl();
      if (!BaseRecordDecl)
        continue;

      checkDestructorLater(BaseRecordDecl);
    }
  }

public:
  bool VisitCallExpr(CallExpr *CE) {
    if (!isa_and_nonnull<FunctionDecl>(CE->getCalleeDecl()))
      return true;

    const auto *FD = dyn_cast<FunctionDecl>(CE->getCalleeDecl());

    if (const auto *DD = dyn_cast<CXXDestructorDecl>(FD)) {
      const CXXRecordDecl *Parent = DD->getParent();
      checkDestructorLater(Parent);
      return true;
    }

    checkFunctionLater(FD);
    return true;
  }

  bool VisitVarDecl(VarDecl *VD) {
    if (VD->isStaticLocal())
      return true;

    const Type *DT = VD->getType().getTypePtrOrNull();
    if (!DT)
      return true;

    const CXXRecordDecl *DeleteDecl = DT->getAsCXXRecordDecl();
    if (!DeleteDecl)
      return true;

    checkDestructorLater(DeleteDecl);
    return true;
  }

  bool VisitCXXConstructExpr(CXXConstructExpr *CE) {
    const CXXConstructorDecl *CD = CE->getConstructor();

    checkFunctionLater(static_cast<const FunctionDecl *>(CD));

    // If we are traversing a function, then all the temporary and non-temporary
    // objects will have their destructor called at the end of the scope. So
    // better traverse the destructors as well.
    if (isInFunction()) {
      const CXXRecordDecl *RD = CD->getParent();
      checkDestructorLater(RD);
    }
    return true;
  }

  bool VisitCXXDeleteExpr(CXXDeleteExpr *DE) {
    const Type *DeleteType = DE->getDestroyedType().getTypePtrOrNull();
    if (!DeleteType)
      return true;

    const CXXRecordDecl *DeleteDecl = DeleteType->getAsCXXRecordDecl();
    if (!DeleteDecl)
      return true;

    checkDestructorLater(DeleteDecl);
    return true;
  }

  bool VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *UE) {
    return false;
  }
  bool VisitOffsetOfExpr(OffsetOfExpr *OE) { return false; }

private:
  void traverseFunctionsToBeChecked() {
    IsInFunction = true;

    // We could find more functions to be checked while checking functions.
    // Because a simple iterator could get invalidated, we index into the array.
    for (size_t I = 0; I < FunctionsToBeChecked.size(); ++I) {
      const FunctionDecl *Func = FunctionsToBeChecked[I];

      if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(Func))
        for (const CXXCtorInitializer *Init : Constructor->inits())
          RecursiveASTVisitor<T>::TraverseStmt(Init->getInit());

      // Look at the function parameters as well. They get destroyed at the end
      // of the scope.
      for (const ParmVarDecl *Param : Func->parameters()) {
        const Type *ParamType = Param->getType().getTypePtrOrNull();
        if (!ParamType)
          continue;

        const CXXRecordDecl *TypeDecl = ParamType->getAsCXXRecordDecl();
        if (!TypeDecl)
          continue;

        checkDestructorLater(TypeDecl);
      }

      // The hasBody check should happen before we add the function to the
      // array.
      assert(Func->hasBody());
      RecursiveASTVisitor<T>::TraverseStmt(Func->getBody());
    }
  }

  // Will be true if we are traversing function/constructor/destructor bodies
  // that can be called from the original starting point of the traversal.
  bool IsInFunction;

  // We check inside functions only if the functions hasn't already been checked
  // during the current traversal. We use this array to check if the function is
  // already registered to be checked.
  llvm::SmallVector<const FunctionDecl *> FunctionsToBeChecked;
};

} // namespace clang::tidy::utils

#endif
