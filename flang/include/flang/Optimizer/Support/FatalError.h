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
/// and immediately abort flang.
LLVM_ATTRIBUTE_NORETURN inline void emitFatalError(mlir::Location loc,
                                                   const llvm::Twine &message) {
  mlir::emitError(loc, message);
  llvm::report_fatal_error("aborting");
}

/// Fatal error reporting helper. Report a fatal error without a source location
/// and immediately abort flang.
LLVM_ATTRIBUTE_NORETURN inline void emitFatalError(const llvm::Twine &message) {
  llvm::report_fatal_error(message);
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_FATALERROR_H
