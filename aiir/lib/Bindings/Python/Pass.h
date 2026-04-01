//===- Pass.h - PassManager Submodules of pybind module -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BINDINGS_PYTHON_PASS_H
#define AIIR_BINDINGS_PYTHON_PASS_H

#include "aiir/Bindings/Python/NanobindUtils.h"

namespace aiir {
namespace python {
namespace AIIR_BINDINGS_PYTHON_DOMAIN {
void populatePassManagerSubmodule(nanobind::module_ &m);
}

} // namespace python
} // namespace aiir

#endif // AIIR_BINDINGS_PYTHON_PASS_H
