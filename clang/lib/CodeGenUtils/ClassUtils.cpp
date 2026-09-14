//===--- ClassUtils.cpp - Shared C++ class emission queries ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/ClassUtils.h"
#include "clang/AST/EvaluatedExprVisitor.h"

namespace clang::CodeGenUtils {
namespace {
/// A visitor which checks whether an initializer uses 'this' in a
/// way which requires the vtable to be properly set.
struct DynamicThisUseChecker
    : ConstEvaluatedExprVisitor<DynamicThisUseChecker> {
  using super = ConstEvaluatedExprVisitor<DynamicThisUseChecker>;

  bool UsesThis = false;

  DynamicThisUseChecker(const ASTContext &C) : super(C) {}

  // Black-list all explicit and implicit references to 'this'.
  //
  // Do we need to worry about external references to 'this' derived
  // from arbitrary code?  If so, then anything which runs arbitrary
  // external code might potentially access the vtable.
  void VisitCXXThisExpr(const CXXThisExpr *E) { UsesThis = true; }
};
} // namespace

bool baseInitializerUsesThis(ASTContext &Ctx, const Expr *Init) {
  DynamicThisUseChecker Checker(Ctx);
  Checker.Visit(Init);
  return Checker.UsesThis;
}

static bool
hasTrivialDestructorBody(ASTContext &Context,
                         const CXXRecordDecl *BaseClassDecl,
                         const CXXRecordDecl *MostDerivedClassDecl) {
  // If the destructor is trivial we don't have to check anything else.
  if (BaseClassDecl->hasTrivialDestructor())
    return true;

  if (!BaseClassDecl->getDestructor()->hasTrivialBody())
    return false;

  // Check fields.
  for (const auto *Field : BaseClassDecl->fields())
    if (!fieldHasTrivialDestructorBody(Context, Field))
      return false;

  // Check non-virtual bases.
  for (const auto &I : BaseClassDecl->bases()) {
    if (I.isVirtual())
      continue;

    const auto *NonVirtualBase = I.getType()->castAsCXXRecordDecl();
    if (!hasTrivialDestructorBody(Context, NonVirtualBase,
                                  MostDerivedClassDecl))
      return false;
  }

  if (BaseClassDecl == MostDerivedClassDecl) {
    // Check virtual bases.
    for (const auto &I : BaseClassDecl->vbases()) {
      const auto *VirtualBase = I.getType()->castAsCXXRecordDecl();
      if (!hasTrivialDestructorBody(Context, VirtualBase, MostDerivedClassDecl))
        return false;
    }
  }

  return true;
}

bool fieldHasTrivialDestructorBody(ASTContext &Context,
                                   const FieldDecl *Field) {
  QualType FieldBaseElementType = Context.getBaseElementType(Field->getType());

  auto *FieldClassDecl = FieldBaseElementType->getAsCXXRecordDecl();
  if (!FieldClassDecl)
    return true;

  // The destructor for an implicit anonymous union member is never invoked.
  if (FieldClassDecl->isUnion() && FieldClassDecl->isAnonymousStructOrUnion())
    return true;

  return hasTrivialDestructorBody(Context, FieldClassDecl, FieldClassDecl);
}

bool canSkipVTablePointerInitialization(ASTContext &Ctx,
                                        const CXXDestructorDecl *Dtor) {
  const CXXRecordDecl *ClassDecl = Dtor->getParent();
  if (!ClassDecl->isDynamicClass())
    return true;

  // For a final class, the vtable pointer is known to already point to the
  // class's vtable.
  if (ClassDecl->isEffectivelyFinal())
    return true;

  if (!Dtor->hasTrivialBody())
    return false;

  // Check the fields.
  for (const auto *Field : ClassDecl->fields())
    if (!fieldHasTrivialDestructorBody(Ctx, Field))
      return false;

  return true;
}

bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit) {
  const Type *BaseType = BaseInit->getBaseClass();
  return BaseType->castAsCXXRecordDecl()->isDynamicClass();
}

} // namespace clang::CodeGenUtils
