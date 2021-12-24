//===-- CanonicalizationPatterns.td - FIR Canonicalization Patterns -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines pattern rewrites for fir optimizations
///
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FIR_REWRITE_PATTERNS
#define FORTRAN_FIR_REWRITE_PATTERNS

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"
include "flang/Optimizer/Dialect/FIROps.td"

def IdenticalTypePred : Constraint<CPred<"$0.getType() == $1.getType()">>;
def IntegerTypePred : Constraint<CPred<"fir::isa_integer($0.getType())">>;
def IndexTypePred : Constraint<CPred<"$0.getType().isa<mlir::IndexType>()">>;

def SmallerWidthPred
    : Constraint<CPred<"$0.getType().getIntOrFloatBitWidth() "
                       "<= $1.getType().getIntOrFloatBitWidth()">>;

def ConvertConvertOptPattern
    : Pat<(fir_ConvertOp (fir_ConvertOp $arg)),
          (fir_ConvertOp $arg),
          [(IntegerTypePred $arg)]>;

def RedundantConvertOptPattern
    : Pat<(fir_ConvertOp:$res $arg),
          (replaceWithValue $arg),
          [(IdenticalTypePred $res, $arg)
          ,(IntegerTypePred $arg)]>;

def CombineConvertOptPattern
    : Pat<(fir_ConvertOp:$res(fir_ConvertOp:$irm $arg)),
          (replaceWithValue $arg),
          [(IdenticalTypePred $res, $arg)
          ,(IntegerTypePred $arg)
          ,(IntegerTypePred $irm)
          ,(SmallerWidthPred $arg, $irm)]>;

def createConstantOp
    : NativeCodeCall<"$_builder.create<mlir::arith::ConstantOp>"
                     "($_loc, $_builder.getIndexType(), "
                     "rewriter.getIndexAttr($1.dyn_cast<IntegerAttr>()"
                     ".getInt()))">;

def ForwardConstantConvertPattern
    : Pat<(fir_ConvertOp:$res (Arith_ConstantOp:$cnt $attr)),
          (createConstantOp $res, $attr),
          [(IndexTypePred $res)
          ,(IntegerTypePred $cnt)]>;

#endif // FORTRAN_FIR_REWRITE_PATTERNS
