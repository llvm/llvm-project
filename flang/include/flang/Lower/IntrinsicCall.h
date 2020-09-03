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

/// Generate the FIR+MLIR operations for the generic intrinsic \p name
/// with arguments \p args and expected result type \p resultType.
/// Returned mlir::Value is the returned Fortran intrinsic value.
fir::ExtendedValue genIntrinsicCall(FirOpBuilder &, mlir::Location,
                                    llvm::StringRef name, mlir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args);

/// Get SymbolRefAttr of runtime (or wrapper function containing inlined
// implementation) of an unrestricted intrinsic (defined by its signature
// and generic name)
mlir::SymbolRefAttr
getUnrestrictedIntrinsicSymbolRefAttr(FirOpBuilder &, mlir::Location,
                                      llvm::StringRef name,
                                      mlir::FunctionType signature);

//===--------------------------------------------------------------------===//
// Direct access to intrinsics that may be used by lowering outside
// of intrinsic call lowering.
//===--------------------------------------------------------------------===//

/// Generate maximum. There must be at least one argument and all arguments
/// must have the same type.
mlir::Value genMax(FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate minimum. Same constraints as genMax.
mlir::Value genMin(FirOpBuilder &, mlir::Location,
                   llvm::ArrayRef<mlir::Value> args);

/// Generate power function x**y with given the expected
/// result type.
mlir::Value genPow(FirOpBuilder &, mlir::Location, mlir::Type resultType,
                   mlir::Value x, mlir::Value y);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_INTRINSICCALL_H
