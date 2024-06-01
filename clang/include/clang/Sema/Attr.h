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
#include "clang/AST/DeclObjC.h"
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

} // namespace clang
#endif // LLVM_CLANG_SEMA_ATTR_H
