//===- LinalgPasses.cpp - Pybind module for the Linalg passes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Linalg.h"

#include "aiir/Bindings/Python/Nanobind.h"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

NB_MODULE(_aiirLinalgPasses, m) {
  m.doc() = "AIIR Linalg Dialect Passes";

  // Register all Linalg passes on load.
  aiirRegisterLinalgPasses();
}
