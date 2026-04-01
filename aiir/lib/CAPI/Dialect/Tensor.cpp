//===- Tensor.cpp - C Interface for Tensor dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tensor/IR/Tensor.h"
#include "aiir-c/Dialect/Tensor.h"
#include "aiir/CAPI/Registration.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Tensor, tensor,
                                      aiir::tensor::TensorDialect)
