//===--- DialectIRDL.cpp - Pybind module for IRDL dialect API support ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/IRDL.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/Bindings/Python/IRCore.h"
#include "aiir/Bindings/Python/Nanobind.h"
#include "aiir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;
using namespace aiir::python::AIIR_BINDINGS_PYTHON_DOMAIN;
using namespace aiir::python::nanobind_adaptors;

static void populateDialectIRDLSubmodule(nb::module_ &m) {
  m.def(
      "load_dialects",
      [](PyModule &module) {
        if (aiirLogicalResultIsFailure(aiirLoadIRDLDialects(module.get())))
          throw std::runtime_error(
              "failed to load IRDL dialects from the input module");
      },
      nb::arg("module"), "Load IRDL dialects from the given module.");
}

NB_MODULE(_aiirDialectsIRDL, m) {
  m.doc() = "AIIR IRDL dialect.";

  populateDialectIRDLSubmodule(m);
}
