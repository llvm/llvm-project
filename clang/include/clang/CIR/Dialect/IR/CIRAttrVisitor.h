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

template <typename ImplClass, typename RetTy> class CirAttrVisitor {
public:
  // FIXME: Create a TableGen list to automatically handle new attributes
  RetTy visit(mlir::Attribute attr) {
    if (const auto intAttr = mlir::dyn_cast<cir::IntAttr>(attr))
      return getImpl().visitCirIntAttr(intAttr);
    if (const auto fltAttr = mlir::dyn_cast<cir::FPAttr>(attr))
      return getImpl().visitCirFPAttr(fltAttr);
    if (const auto ptrAttr = mlir::dyn_cast<cir::ConstPtrAttr>(attr))
      return getImpl().visitCirConstPtrAttr(ptrAttr);
    llvm_unreachable("unhandled attribute type");
  }

  // If the implementation chooses not to implement a certain visit
  // method, fall back to the parent.
  RetTy visitCirIntAttr(cir::IntAttr attr) {
    return getImpl().visitCirAttr(attr);
  }
  RetTy visitCirFPAttr(cir::FPAttr attr) {
    return getImpl().visitCirAttr(attr);
  }
  RetTy visitCirConstPtrAttr(cir::ConstPtrAttr attr) {
    return getImpl().visitCirAttr(attr);
  }

  RetTy visitCirAttr(mlir::Attribute attr) { return RetTy(); }

  ImplClass &getImpl() { return *static_cast<ImplClass *>(this); }
};

} // namespace cir

#endif // LLVM_CLANG_CIR_DIALECT_IR_CIRATTRVISITOR_H
