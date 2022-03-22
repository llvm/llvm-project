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
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticOps.td"
include "flang/Optimizer/Dialect/FIROps.td"

def IdenticalTypePred : Constraint<CPred<"$0.getType() == $1.getType()">>;
def IntegerTypePred : Constraint<CPred<"fir::isa_integer($0.getType())">>;
def IndexTypePred : Constraint<CPred<"$0.getType().isa<mlir::IndexType>()">>;

// Widths are monotonic.
//   $0.bits >= $1.bits >= $2.bits or $0.bits <= $1.bits <= $2.bits
def MonotonicTypePred
    : Constraint<CPred<"(($0.getType().isa<mlir::IntegerType>() && "
                       "  $1.getType().isa<mlir::IntegerType>() && "
                       "  $2.getType().isa<mlir::IntegerType>()) || "
                       " ($0.getType().isa<mlir::FloatType>() && "
                       "  $1.getType().isa<mlir::FloatType>() && "
                       "  $2.getType().isa<mlir::FloatType>())) && "
                       "(($0.getType().getIntOrFloatBitWidth() <= "
                       "  $1.getType().getIntOrFloatBitWidth() && "
                       "  $1.getType().getIntOrFloatBitWidth() <= "
                       "  $2.getType().getIntOrFloatBitWidth()) || "
                       " ($0.getType().getIntOrFloatBitWidth() >= "
                       "  $1.getType().getIntOrFloatBitWidth() && "
                       "  $1.getType().getIntOrFloatBitWidth() >= "
                       "  $2.getType().getIntOrFloatBitWidth()))">>;

def IntPred : Constraint<CPred<
                       "$0.getType().isa<mlir::IntegerType>() && "
                       "$1.getType().isa<mlir::IntegerType>()">>;
                       
// If both are int type and the first is smaller than the second.
//   $0.bits <= $1.bits
def SmallerWidthPred : Constraint<CPred<
                       "$0.getType().getIntOrFloatBitWidth() <= "
                       "$1.getType().getIntOrFloatBitWidth()">>;
def StrictSmallerWidthPred : Constraint<CPred<
                       "$0.getType().getIntOrFloatBitWidth() < "
                       "$1.getType().getIntOrFloatBitWidth()">>;

// floats or ints that undergo successive extensions or successive truncations.
def ConvertConvertOptPattern
    : Pat<(fir_ConvertOp:$res (fir_ConvertOp:$irm $arg)),
          (fir_ConvertOp $arg),
          [(MonotonicTypePred $res, $irm, $arg)]>;

// Widths are increasingly monotonic to type index, so there is no
// possibility of a truncation before the conversion to index.
//   $res == index && $irm.bits >= $arg.bits
def ConvertAscendingIndexOptPattern
    : Pat<(fir_ConvertOp:$res (fir_ConvertOp:$irm $arg)),
          (fir_ConvertOp $arg),
          [(IndexTypePred $res), (IntPred $irm, $arg),
           (SmallerWidthPred $arg, $irm)]>;

// Widths are decreasingly monotonic from type index, so the truncations
// continue to lop off more bits.
//   $arg == index && $res.bits < $irm.bits
def ConvertDescendingIndexOptPattern
    : Pat<(fir_ConvertOp:$res (fir_ConvertOp:$irm $arg)),
          (fir_ConvertOp $arg),
          [(IndexTypePred $arg), (IntPred $irm, $res),
           (SmallerWidthPred $res, $irm)]>;

// Useless convert to exact same type.
def RedundantConvertOptPattern
    : Pat<(fir_ConvertOp:$res $arg),
          (replaceWithValue $arg),
          [(IdenticalTypePred $res, $arg)]>;

// Useless extension followed by truncation to get same width integer.
def CombineConvertOptPattern
    : Pat<(fir_ConvertOp:$res(fir_ConvertOp:$irm $arg)),
          (replaceWithValue $arg),
          [(IntPred $res, $arg), (IdenticalTypePred $res, $arg),
           (IntPred $arg, $irm), (SmallerWidthPred $arg, $irm)]>;

// Useless extension followed by truncation to get smaller width integer.
def CombineConvertTruncOptPattern
    : Pat<(fir_ConvertOp:$res(fir_ConvertOp:$irm $arg)),
          (fir_ConvertOp $arg),
          [(IntPred $res, $arg), (StrictSmallerWidthPred $res, $arg),
           (IntPred $arg, $irm), (SmallerWidthPred $arg, $irm)]>;

def createConstantOp
    : NativeCodeCall<"$_builder.create<mlir::arith::ConstantOp>"
                     "($_loc, $_builder.getIndexType(), "
                     "rewriter.getIndexAttr($1.dyn_cast<mlir::IntegerAttr>()"
                     ".getInt()))">;

def ForwardConstantConvertPattern
    : Pat<(fir_ConvertOp:$res (Arith_ConstantOp:$cnt $attr)),
          (createConstantOp $res, $attr),
          [(IndexTypePred $res), (IntegerTypePred $cnt)]>;

#endif // FORTRAN_FIR_REWRITE_PATTERNS
