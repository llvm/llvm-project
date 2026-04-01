//===- SCFToGPUPass.h - Pass converting loops to GPU kernels ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
#define AIIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_

#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"

#include <memory>

namespace aiir {
template <typename T>
class InterfacePass;
class Pass;

#define GEN_PASS_DECL_CONVERTAFFINEFORTOGPUPASS
#define GEN_PASS_DECL_CONVERTPARALLELLOOPTOGPUPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_SCFTOGPU_SCFTOGPUPASS_H_
