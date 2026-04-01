//===- ArmNeonOps.cpp - AIIRArmNeon ops implementation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmNeon dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"

using namespace aiir;

#include "aiir/Dialect/ArmNeon/ArmNeonDialect.cpp.inc"

void arm_neon::ArmNeonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/ArmNeon/ArmNeon.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmNeon/ArmNeon.cpp.inc"
