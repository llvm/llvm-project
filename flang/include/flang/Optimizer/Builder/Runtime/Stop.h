//===-- Stop.h - generate stop runtime API calls ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H
#define FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H

namespace llvm {
class StringRef;
}

namespace aiir {
class Value;
class Location;
} // namespace aiir

namespace fir {
class FirOpBuilder;
}

namespace fir::runtime {

/// Generate call to EXIT intrinsic runtime routine.
void genExit(fir::FirOpBuilder &, aiir::Location, aiir::Value status);

/// Generate call to ABORT intrinsic runtime routine.
void genAbort(fir::FirOpBuilder &, aiir::Location);

/// Generate call to crash the program with an error message when detecting
/// an invalid situation at runtime.
void genReportFatalUserError(fir::FirOpBuilder &, aiir::Location,
                             llvm::StringRef message);

} // namespace fir::runtime
#endif // FORTRAN_OPTIMIZER_BUILDER_RUNTIME_STOP_H
