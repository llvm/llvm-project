//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
using namespace clang;

#include "clang/AST/AttrImpl.inc"

// FIXME: this should be auto generated from Attr.td
bool Attr::compare(const Attr *A, const Attr *B) {
  if (A->getKind() != B->getKind())
    return A->getKind() < B->getKind();

  switch (A->getKind()) {
  case attr::ObjCBridge: {
    auto *MA = cast<ObjCBridgeAttr>(A);
    auto *MB = cast<ObjCBridgeAttr>(B);
    if (!MA->getBridgedType())
      return true;
    if (!MB->getBridgedType())
      return false;
    return MA->getBridgedType()->getName() < MB->getBridgedType()->getName();
  }
  case attr::ObjCBridgeMutable: {
    auto *MA = cast<ObjCBridgeMutableAttr>(A);
    auto *MB = cast<ObjCBridgeMutableAttr>(B);
    if (!MA->getBridgedType())
      return true;
    if (!MB->getBridgedType())
      return false;
    return MA->getBridgedType()->getName() < MB->getBridgedType()->getName();
  }
  default:
    llvm_unreachable("Not implemented");
  }
  return false;
}
