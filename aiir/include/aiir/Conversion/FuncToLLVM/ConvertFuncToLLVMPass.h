//===- ConvertFuncToLLVMPass.h - Pass entrypoint ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
#define AIIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_

#include <memory>
#include <string>

namespace aiir {
class Pass;

#define GEN_PASS_DECL_CONVERTFUNCTOLLVMPASS
#define GEN_PASS_DECL_SETLLVMMODULEDATALAYOUTPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_FUNCTOLLVM_CONVERTFUNCTOLLVMPASS_H_
