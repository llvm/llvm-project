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
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "flang/Optimizer/Dialect/FIROps.td"

def IdenticalTypePred : Constraint<CPred<"$0.getType() == $1.getType()">>;
def IntegerTypePred : Constraint<CPred<"fir::isa_integer($0.getType())">>;
def IndexTypePred : Constraint<CPred<
                       "mlir::isa<mlir::IndexType>($0.getType())">>;

// Widths are monotonic.
//   $0.bits >= $1.bits >= $2.bits or $0.bits <= $1.bits <= $2.bits
def MonotonicTypePred
    : Constraint<CPred<"((mlir::isa<mlir::IntegerType>($0.getType()) && "
                       "  mlir::isa<mlir::IntegerType>($1.getType()) && "
                       "  mlir::isa<mlir::IntegerType>($2.getType()) && "
                       "  $0.getType().isUnsignedInteger() == $1.getType().isUnsignedInteger() && "
                       "  $1.getType().isUnsignedInteger() == $2.getType().isUnsignedInteger()) || "
                       " (mlir::isa<mlir::FloatType>($0.getType()) && "
                       "  mlir::isa<mlir::FloatType>($1.getType()) && "
                       "  mlir::isa<mlir::FloatType>($2.getType()))) && "
                       "(($0.getType().getIntOrFloatBitWidth() <= "
                       "  $1.getType().getIntOrFloatBitWidth() && "
                       "  $1.getType().getIntOrFloatBitWidth() <= "
                       "  $2.getType().getIntOrFloatBitWidth()) || "
                       " ($0.getType().getIntOrFloatBitWidth() >= "
                       "  $1.getType().getIntOrFloatBitWidth() && "
                       "  $1.getType().getIntOrFloatBitWidth() >= "
                       "  $2.getType().getIntOrFloatBitWidth()))">>;

def IntPred : Constraint<CPred<
                       "mlir::isa<mlir::IntegerType>($0.getType()) && "
                       "mlir::isa<mlir::IntegerType>($1.getType())">>;
                       
// If both are int type and the first is smaller than the second.
//   $0.bits <= $1.bits
def SmallerWidthPred : Constraint<CPred<
                       "$0.getType().getIntOrFloatBitWidth() <= "
                       "$1.getType().getIntOrFloatBitWidth()">>;
def StrictSmallerWidthPred : Constraint<CPred<
                       "$0.getType().getIntOrFloatBitWidth() < "
                       "$1.getType().getIntOrFloatBitWidth()">>;

def PointerCompatiblePred
    : Constraint<CPred<"fir::ConvertOp::isPointerCompatible($0.getType())">>;

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
                     "rewriter.getIndexAttr("
                     "mlir::dyn_cast<mlir::IntegerAttr>($1).getInt()))">;

def ForwardConstantConvertPattern
    : Pat<(fir_ConvertOp:$res (Arith_ConstantOp:$cnt $attr)),
          (createConstantOp $res, $attr),
          [(IndexTypePred $res), (IntegerTypePred $cnt)]>;

// Optimize redundant pointer conversions, e.g.:
// %1 = fir.convert %0 :
//     (!fir.heap<!fir.array<2xf32>>) -> !fir.ref<!fir.array<2xf32>>
// %2 = fir.convert %1 :
//     (!fir.ref<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>>
// Will be optimized into:
// %2 = fir.convert %0 :
//     (!fir.heap<!fir.array<2xf32>>) -> !fir.heap<!fir.array<2xf32>>
// which is redundant due to RedundantConvertOptPattern.
def ChainedPointerConvertsPattern
    : Pat<(fir_ConvertOp:$res(fir_ConvertOp:$irm $arg)), (fir_ConvertOp $arg),
          [(PointerCompatiblePred $arg), (PointerCompatiblePred $irm),
           (PointerCompatiblePred $res)]>;

#endif // FORTRAN_FIR_REWRITE_PATTERNS
