//===-- RegisterOpenMPExtensions.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenMP extensions as applied to FIR dialect.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenMP/Support/RegisterOpenMPExtensions.h"

namespace fir::omp {
void registerOpenMPExtensions(mlir::DialectRegistry &registry) {
  registerAttrsExtensions(registry);
  registerOpInterfacesExtensions(registry);
}

} // namespace fir::omp
