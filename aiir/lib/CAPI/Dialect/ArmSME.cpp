//===- ArmSME.cpp - C Interface for ArmSME dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/ArmSME.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/ArmSME/IR/ArmSME.h"

using namespace aiir::arm_sme;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(ArmSME, arm_sme,
                                      aiir::arm_sme::ArmSMEDialect)
