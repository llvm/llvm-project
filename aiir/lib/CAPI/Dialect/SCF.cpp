//===- SCF.cpp - C Interface for SCF dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir-c/Dialect/SCF.h"
#include "aiir/CAPI/Registration.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(SCF, scf, aiir::scf::SCFDialect)
