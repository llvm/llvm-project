//===- ArmSVEDialect.cpp - AIIR ArmSVE dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSVE dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/TypeUtilities.h"

using namespace aiir;
using namespace aiir::arm_sve;

//===----------------------------------------------------------------------===//
// ScalableVector versions of general helpers for comparison ops
//===----------------------------------------------------------------------===//

/// Return the scalable vector of the same shape and containing i1.
static Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto sVectorType = llvm::dyn_cast<VectorType>(type))
    return VectorType::get(sVectorType.getShape(), i1Type,
                           sVectorType.getScalableDims());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.cpp.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSVE/IR/ArmSVE.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/ArmSVE/IR/ArmSVETypes.cpp.inc"

void ArmSVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/ArmSVE/IR/ArmSVE.cpp.inc"
      >();
}
