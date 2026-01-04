//===- Timing.h - Execution time measurement facilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Facilities to measure and provide statistics on execution time.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_TIMING_H
#define FORTRAN_SUPPORT_TIMING_H

#include "mlir/Support/Timing.h"

namespace Fortran::support {

/// Create a strategy to render the captured times in plain text. This is
/// intended to be passed to a TimingManager.
std::unique_ptr<mlir::OutputStrategy> createTimingFormatterText(
    llvm::raw_ostream &os);

} // namespace Fortran::support

#endif // FORTRAN_SUPPORT_TIMING_H
