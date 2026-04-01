//===- AMDGPUAttrs.cpp - AIIR AMDGPU dialect attributes -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGPU dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/AMDGPU/IR/AMDGPUAttrs.cpp.inc"

void aiir::amdgpu::AMDGPUDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/AMDGPU/IR/AMDGPUAttrs.cpp.inc"
      >();
}
