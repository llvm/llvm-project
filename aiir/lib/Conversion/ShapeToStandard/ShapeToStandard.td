//==-- ShapeToStandard.td - Shape to Standard Patterns -------*- tablegen -*==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower Shape ops to Std.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_SHAPETOSTANDARD_TD
#define AIIR_CONVERSION_SHAPETOSTANDARD_TD

include "aiir/IR/PatternBase.td"
include "aiir/Dialect/Shape/IR/ShapeOps.td"

def BroadcastableStringAttr : NativeCodeCall<[{
  $_builder.getStringAttr("required broadcastable shapes")
}]>;

def CstrBroadcastableToRequire : Pat<(Shape_CstrBroadcastableOp $shapes),
            (Shape_CstrRequireOp
              (Shape_IsBroadcastableOp $shapes),
              (BroadcastableStringAttr))>;

def EqStringAttr : NativeCodeCall<[{
  $_builder.getStringAttr("required equal shapes")
}]>;

def CstrEqToRequire : Pat<(Shape_CstrEqOp $shapes),
  (Shape_CstrRequireOp (Shape_ShapeEqOp $shapes), (EqStringAttr))>;

#endif // AIIR_CONVERSION_SHAPETOSTANDARD_TD
