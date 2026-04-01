//===- Passes.h - Reducer Pass Construction and Registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef AIIR_REDUCER_PASSES_H
#define AIIR_REDUCER_PASSES_H

#include "aiir/Pass/Pass.h"

namespace aiir {

#define GEN_PASS_DECL
#include "aiir/Reducer/Passes.h.inc"

/// Generate the code for registering reducer passes.
#define GEN_PASS_REGISTRATION
#include "aiir/Reducer/Passes.h.inc"

} // namespace aiir

#endif // AIIR_REDUCER_PASSES_H
