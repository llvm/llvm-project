//==--- CodeGenUtils.cpp - Shared Classic CodeGen/CIR CodeGen Utils--C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CodeGenUtils/CodeGenUtils.h"
#include "clang/Basic/TargetInfo.h"

namespace clang::CodeGenUtils {
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

/// Check whether we need to initialize any vtable pointers before calling this
/// destructor.
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
bool hasUnwindExceptions(const LangOptions &LangOpts) {
  // If exceptions are completely disabled, obviously this is false.
  if (!LangOpts.Exceptions)
    return false;

  // If C++ exceptions are enabled, this is true.
  if (LangOpts.CXXExceptions)
    return true;

  // If ObjC exceptions are enabled, this depends on the ABI.
  if (LangOpts.ObjCExceptions) {
    return LangOpts.ObjCRuntime.hasUnwindExceptions();
  }

  return true;
}

bool isAAPCS(const TargetInfo &TargetInfo) {
  return TargetInfo.getABI().starts_with("aapcs");
}
bool isInitializerOfDynamicClass(const CXXCtorInitializer *BaseInit) {
  const Type *BaseType = BaseInit->getBaseClass();
  return BaseType->castAsCXXRecordDecl()->isDynamicClass();
}

} // namespace clang::CodeGenUtils
