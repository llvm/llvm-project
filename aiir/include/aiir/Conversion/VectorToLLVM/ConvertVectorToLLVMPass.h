//===- ConvertVectorToLLVMPass.h - Pass to check Vector->LLVM --- --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_
#define AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_

#include "aiir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "aiir/Dialect/Vector/Transforms/VectorTransforms.h"

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTVECTORTOLLVMPASS
#include "aiir/Conversion/Passes.h.inc"
} // namespace aiir
#endif // AIIR_CONVERSION_VECTORTOLLVM_CONVERTVECTORTOLLVMPASS_H_
