//===- ArmNeon.cpp - C Interface for ArmNeon dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/ArmNeon.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/ArmNeon/ArmNeonDialect.h"

using namespace aiir::arm_neon;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmNeon, arm_neon,
                                      aiir::arm_neon::ArmNeonDialect)
