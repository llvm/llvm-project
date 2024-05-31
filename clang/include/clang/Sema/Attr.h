//===----- Attr.h --- Helper functions for attribute handling in Sema -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file provides helpers for Sema functions that handle attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_ATTR_H
#define LLVM_CLANG_SEMA_ATTR_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/Casting.h"

namespace clang {

/// isFuncOrMethodForAttrSubject - Return true if the given decl has function
/// type (function or function-typed variable) or an Objective-C
/// method.
inline bool isFuncOrMethodForAttrSubject(const Decl *D) {
  return (D->getFunctionType() != nullptr) || llvm::isa<ObjCMethodDecl>(D);
}

/// Return true if the given decl has function type (function or
/// function-typed variable) or an Objective-C method or a block.
inline bool isFunctionOrMethodOrBlockForAttrSubject(const Decl *D) {
  return isFuncOrMethodForAttrSubject(D) || llvm::isa<BlockDecl>(D);
}

/// Return true if the given decl has a declarator that should have
/// been processed by Sema::GetTypeForDeclarator.
inline bool hasDeclarator(const Decl *D) {
  // In some sense, TypedefDecl really *ought* to be a DeclaratorDecl.
  return isa<DeclaratorDecl>(D) || isa<BlockDecl>(D) || isa<TypedefNameDecl>(D) ||
         isa<ObjCPropertyDecl>(D);
}

/// hasFunctionProto - Return true if the given decl has a argument
/// information. This decl should have already passed
/// isFuncOrMethodForAttrSubject or isFunctionOrMethodOrBlockForAttrSubject.
inline bool hasFunctionProto(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return isa<FunctionProtoType>(FnTy);
  return isa<ObjCMethodDecl>(D) || isa<BlockDecl>(D);
}

/// getFunctionOrMethodNumParams - Return number of function or method
/// parameters. It is an error to call this on a K&R function (use
/// hasFunctionProto first).
inline unsigned getFunctionOrMethodNumParams(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->getNumParams();
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getNumParams();
  return cast<ObjCMethodDecl>(D)->param_size();
}

inline const ParmVarDecl *getFunctionOrMethodParam(const Decl *D,
                                                   unsigned Idx) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getParamDecl(Idx);
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getParamDecl(Idx);
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx);
  return nullptr;
}

inline QualType getFunctionOrMethodParamType(const Decl *D, unsigned Idx) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->getParamType(Idx);
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->getParamDecl(Idx)->getType();

  return cast<ObjCMethodDecl>(D)->parameters()[Idx]->getType();
}

inline SourceRange getFunctionOrMethodParamRange(const Decl *D, unsigned Idx) {
  if (auto *PVD = getFunctionOrMethodParam(D, Idx))
    return PVD->getSourceRange();
  return SourceRange();
}

inline QualType getFunctionOrMethodResultType(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return FnTy->getReturnType();
  return cast<ObjCMethodDecl>(D)->getReturnType();
}

inline SourceRange getFunctionOrMethodResultSourceRange(const Decl *D) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    return FD->getReturnTypeSourceRange();
  if (const auto *MD = dyn_cast<ObjCMethodDecl>(D))
    return MD->getReturnTypeSourceRange();
  return SourceRange();
}

inline bool isFunctionOrMethodVariadic(const Decl *D) {
  if (const FunctionType *FnTy = D->getFunctionType())
    return cast<FunctionProtoType>(FnTy)->isVariadic();
  if (const auto *BD = dyn_cast<BlockDecl>(D))
    return BD->isVariadic();
  return cast<ObjCMethodDecl>(D)->isVariadic();
}

inline bool isInstanceMethod(const Decl *D) {
  if (const auto *MethodDecl = dyn_cast<CXXMethodDecl>(D))
    return MethodDecl->isInstance();
  return false;
}

} // namespace clang
#endif // LLVM_CLANG_SEMA_ATTR_H
