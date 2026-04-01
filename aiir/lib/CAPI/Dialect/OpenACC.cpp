//===- OpenACC.cpp - C Interface for OpenACC dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/OpenACC.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"

using namespace aiir::acc;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(OpenACC, acc, aiir::acc::OpenACCDialect)
