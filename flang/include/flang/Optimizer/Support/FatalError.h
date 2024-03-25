//===-- Optimizer/Support/FatalError.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H
#define FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/ErrorHandling.h"

namespace fir {

/// Fatal error reporting helper. Report a fatal error with a source location
/// and immediately interrupt flang. If `genCrashDiag` is true, then
/// the execution is aborted and the backtrace is printed, otherwise,
/// flang exits with non-zero exit code and without backtrace printout.
[[noreturn]] inline void emitFatalError(mlir::Location loc,
                                        const llvm::Twine &message,
                                        bool genCrashDiag = true) {
  mlir::emitError(loc, message);
  llvm::report_fatal_error("aborting", genCrashDiag);
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H
