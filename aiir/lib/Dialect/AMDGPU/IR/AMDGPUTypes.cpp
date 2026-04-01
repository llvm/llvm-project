//===- AMDGPUTypes.cpp - AIIR AMDGPU dialect types ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect types.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"

void aiir::amdgpu::AMDGPUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aiir/Dialect/AMDGPU/IR/AMDGPUTypes.cpp.inc"
      >();
}
