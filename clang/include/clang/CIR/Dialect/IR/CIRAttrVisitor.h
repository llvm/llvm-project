//===- CIRAttrVisitor.h - Visitor for CIR attributes ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CirAttrVisitor interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H
#define LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H

#include "clang/CIR/Dialect/IR/CIRAttrs.h"

namespace cir {

#define DISPATCH(NAME) return getImpl()->visitCir##NAME(cirAttr);

template <typename ImplClass, typename RetTy> class CirAttrVisitor {
public:
  RetTy visit(mlir::Attribute attr) {
#define ATTRDEF(NAME)                                                          \
  if (const auto cirAttr = mlir::dyn_cast<cir::NAME>(attr))                    \
    DISPATCH(NAME);
#include "clang/CIR/Dialect/IR/CIRAttrDefsList.inc"
    llvm_unreachable("unhandled attribute type");
  }

  // If the implementation chooses not to implement a certain visit
  // method, fall back to the parent.
#define ATTRDEF(NAME)                                                          \
  RetTy visitCir##NAME(NAME cirAttr) { DISPATCH(Attr); }
#include "clang/CIR/Dialect/IR/CIRAttrDefsList.inc"

  RetTy visitCirAttr(mlir::Attribute attr) { return RetTy(); }

  ImplClass *getImpl() { return static_cast<ImplClass *>(this); }
};

#undef DISPATCH

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H
