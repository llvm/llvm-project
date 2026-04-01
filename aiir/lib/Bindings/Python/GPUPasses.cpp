//===- GPUPasses.cpp - Pybind module for the GPU passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "aiir-c/Dialect/GPU.h"

#include "aiir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_aiirGPUPasses, m) {
  m.doc() = "AIIR GPU Dialect Passes";

  // Register all GPU passes on load.
  aiirRegisterGPUPasses();
}
