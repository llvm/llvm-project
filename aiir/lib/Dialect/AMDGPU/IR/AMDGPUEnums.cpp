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
#include "llvm/ADT/StringExtras.h"

#include "aiir/Dialect/AMDGPU/IR/AMDGPUEnums.cpp.inc"
