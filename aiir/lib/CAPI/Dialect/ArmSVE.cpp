//===- ArmSVE.cpp - C Interface for ArmSVE dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/ArmSVE.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/ArmSVE/IR/ArmSVEDialect.h"

using namespace aiir::arm_sve;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmSVE, arm_sve,
                                      aiir::arm_sve::ArmSVEDialect)
