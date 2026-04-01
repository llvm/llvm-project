//===- MathDialect.cpp - AIIR dialect for Math implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Dialect/Math/IR/Math.h"
#include "aiir/Transforms/InliningUtils.h"

using namespace aiir;
using namespace aiir::math;

#include "aiir/Dialect/Math/IR/MathOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void aiir::math::MathDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/Math/IR/MathOps.cpp.inc"
      >();
  addInterfaces<MathInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, MathDialect>();
}
