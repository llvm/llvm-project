//===- EmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/Transforms/Passes.h"

#include "aiir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Func/Transforms/FuncConversions.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/TypeUtilities.h"
#include "aiir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace aiir;

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

arith::NarrowTypeEmulationConverter::NarrowTypeEmulationConverter(
    unsigned targetBitwidth)
    : loadStoreBitwidth(targetBitwidth) {
  assert(llvm::isPowerOf2_32(targetBitwidth) &&
         "Only power-of-two integers are supported");

  // Allow unknown types.
  addConversion([](Type ty) -> std::optional<Type> { return ty; });

  // Function case.
  addConversion([this](FunctionType ty) -> std::optional<Type> {
    SmallVector<Type> inputs;
    if (failed(convertTypes(ty.getInputs(), inputs)))
      return nullptr;

    SmallVector<Type> results;
    if (failed(convertTypes(ty.getResults(), results)))
      return nullptr;

    return FunctionType::get(ty.getContext(), inputs, results);
  });
}

void arith::populateArithNarrowTypeEmulationPatterns(
    const NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}
