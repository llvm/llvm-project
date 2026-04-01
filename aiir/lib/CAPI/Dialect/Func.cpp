//===- Func.cpp - C Interface for Func dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Func.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(Func, func, aiir::func::FuncDialect)

void aiirFuncSetArgAttr(AiirOperation op, intptr_t pos, AiirStringRef name,
                        AiirAttribute attr) {
  llvm::cast<aiir::func::FuncOp>(unwrap(op))
      .setArgAttr(pos, unwrap(name), unwrap(attr));
}

void aiirFuncSetResultAttr(AiirOperation op, intptr_t pos, AiirStringRef name,
                           AiirAttribute attr) {
  llvm::cast<aiir::func::FuncOp>(unwrap(op))
      .setResultAttr(pos, unwrap(name), unwrap(attr));
}
