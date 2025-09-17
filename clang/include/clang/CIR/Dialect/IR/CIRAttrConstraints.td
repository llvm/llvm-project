//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CIR dialect attributes constraints.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_IR_CIRATTRCONSTRAINTS_TD
#define CLANG_CIR_DIALECT_IR_CIRATTRCONSTRAINTS_TD

include "mlir/IR/CommonAttrConstraints.td"

class CIR_IsAttrPred<code attr> : CPred<"::mlir::isa<" # attr # ">($_self)">;

class CIR_AttrConstraint<code attr, string summary = "">
    : Attr<CIR_IsAttrPred<attr>, summary>;

//===----------------------------------------------------------------------===//
// IntAttr constraints
//===----------------------------------------------------------------------===//

def CIR_AnyIntAttr : CIR_AttrConstraint<"::cir::IntAttr", "integer attribute">;

//===----------------------------------------------------------------------===//
// FPAttr constraints
//===----------------------------------------------------------------------===//

def CIR_AnyFPAttr : CIR_AttrConstraint<"::cir::FPAttr",
  "floating-point attribute">;

def CIR_AnyIntOrFloatAttr : AnyAttrOf<[CIR_AnyIntAttr, CIR_AnyFPAttr],
    "integer or floating point type"> {
  string cppType = "::mlir::TypedAttr";
}

//===----------------------------------------------------------------------===//
// ArrayAttr constraints
//===----------------------------------------------------------------------===//

def CIR_IntArrayAttr : TypedArrayAttrBase<CIR_AnyIntAttr,
   "integer array attribute">;

#endif // CLANG_CIR_DIALECT_IR_CIRATTRCONSTRAINTS_TD