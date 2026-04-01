//===- LoweringOptions.cpp -  Common config for lowering to LLVM ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Conversion/LLVMCommon/LoweringOptions.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/Interfaces/DataLayoutInterfaces.h"

using namespace aiir;

aiir::LowerToLLVMOptions::LowerToLLVMOptions(AIIRContext *ctx)
    : LowerToLLVMOptions(ctx, DataLayout()) {}

aiir::LowerToLLVMOptions::LowerToLLVMOptions(AIIRContext *ctx,
                                             const DataLayout &dl) {
  indexBitwidth = dl.getTypeSizeInBits(IndexType::get(ctx));
}
