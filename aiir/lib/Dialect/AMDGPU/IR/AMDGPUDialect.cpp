//===- AMDGPUDialect.cpp - AIIR AMDGPU dialect implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/ROCDLDialect.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Transforms/InliningUtils.h"

using namespace aiir;
using namespace aiir::amdgpu;

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.cpp.inc"

namespace {
struct AMDGPUInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void AMDGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/AMDGPU/IR/AMDGPU.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
  addInterfaces<AMDGPUInlinerInterface>();
}
