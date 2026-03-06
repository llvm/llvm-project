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
// Exceptions constraints
//===----------------------------------------------------------------------===//

def CIR_AnyCatchAllAttr
    : CIR_AttrConstraint<"::cir::CatchAllAttr", "catch all attribute">;

def CIR_AnyUnwindAttr
    : CIR_AttrConstraint<"::cir::UnwindAttr", "unwind attribute">;

//===----------------------------------------------------------------------===//
// GlobalViewAttr constraints
//===----------------------------------------------------------------------===//

def CIR_AnyGlobalViewAttr
    : CIR_AttrConstraint<"::cir::GlobalViewAttr", "GlobalView attribute">;

def CIR_AnyIntOrGlobalViewAttr
    : AnyAttrOf<[CIR_AnyIntAttr, CIR_AnyGlobalViewAttr],
                "integer or global view attribute"> {
  string cppType = "::mlir::TypedAttr";
}

def CIR_TryHandlerAttr
    : AnyAttrOf<[CIR_AnyGlobalViewAttr, CIR_AnyCatchAllAttr, CIR_AnyUnwindAttr],
                "catch all or unwind or global view attribute"> {
  string cppType = "::mlir::TypedAttr";
}

//===----------------------------------------------------------------------===//
// ArrayAttr constraints
//===----------------------------------------------------------------------===//

def CIR_IntArrayAttr : TypedArrayAttrBase<CIR_AnyIntAttr,
   "integer array attribute">;

def CIR_IntOrGlobalViewArrayAttr : TypedArrayAttrBase<CIR_AnyIntOrGlobalViewAttr,
   "integer or global view array attribute">{
  string cppType = "::mlir::ArrayAttr";
}

def CIR_TryHandlerArrayAttr : TypedArrayAttrBase<CIR_TryHandlerAttr,
                         "catch all or unwind or global view array attribute">;

#endif // CLANG_CIR_DIALECT_IR_CIRATTRCONSTRAINTS_TD
