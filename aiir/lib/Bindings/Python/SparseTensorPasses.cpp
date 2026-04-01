//===- SparseTensorPasses.cpp - Pybind module for the SparseTensor passes -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/SparseTensor.h"

#include "aiir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_aiirSparseTensorPasses, m) {
  m.doc() = "AIIR SparseTensor Dialect Passes";

  // Register all SparseTensor passes on load.
  aiirRegisterSparseTensorPasses();
}
