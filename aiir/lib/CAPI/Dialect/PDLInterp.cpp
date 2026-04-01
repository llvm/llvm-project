//===- PDLInterp.cpp - C Interface for PDLInterp dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/PDLInterp.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/PDLInterp/IR/PDLInterp.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(PDLInterp, pdl_interp,
                                      aiir::pdl_interp::PDLInterpDialect)
