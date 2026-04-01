//===- XeVM.cpp - C Interface for XeVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/XeVM.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/LLVMIR/XeVMDialect.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(XeVM, xevm, aiir::xevm::XeVMDialect)
