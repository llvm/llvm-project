//===-- Lower/IntrinsicCall.h -- lowering of intrinsics ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_INTRINSICCALL_H
#define FORTRAN_LOWER_INTRINSICCALL_H

#include "flang/Lower/FIRBuilder.h"

namespace fir {
class ExtendedValue;
}

namespace Fortran::lower {

// TODO: Expose interface to get specific intrinsic function address.
// TODO: Handle intrinsic subroutine.
// TODO: Intrinsics that do not require their arguments to be defined
//   (e.g shape inquiries) might not fit in the current interface that
//   requires mlir::Value to be provided.
// TODO: Error handling interface ?
// TODO: Implementation is incomplete. Many intrinsics to tbd.

/// Helper for building calls to intrinsic functions in the runtime support
/// libraries.
class IntrinsicCallOpsHelper {
public:
  explicit IntrinsicCallOpsHelper(FirOpBuilder &builder, mlir::Location loc)
      : builder(builder), loc(loc) {}
  IntrinsicCallOpsHelper(const IntrinsicCallOpsHelper &) = delete;

  /// Generate the FIR+MLIR operations for the generic intrinsic \p name
  /// with arguments \p args and expected result type \p resultType.
  /// Returned mlir::Value is the returned Fortran intrinsic value.
  fir::ExtendedValue genIntrinsicCall(llvm::StringRef name,
                                      mlir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args);

  //===--------------------------------------------------------------------===//
  // Direct access to intrinsics that may be used by lowering outside
  // of intrinsic call lowering.
  //===--------------------------------------------------------------------===//

  /// Generate maximum. There must be at least one argument and all arguments
  /// must have the same type.
  mlir::Value genMax(llvm::ArrayRef<mlir::Value> args);

  /// Generate minimum. Same constraints as genMax.
  mlir::Value genMin(llvm::ArrayRef<mlir::Value> args);

  /// Generate power function x**y with given the expected
  /// result type.
  mlir::Value genPow(mlir::Type resultType, mlir::Value x, mlir::Value y);

private:
  FirOpBuilder &builder;
  mlir::Location loc;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_INTRINSICCALL_H
