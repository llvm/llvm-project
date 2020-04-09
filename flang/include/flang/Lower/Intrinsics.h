//===-- Lower/Intrinsics.h -- lowering of intrinsics ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builder routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_INTRINSICS_H_
#define FORTRAN_LOWER_INTRINSICS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace Fortran::lower {

class FirOpBuilder;

/// IntrinsicLibrary generates FIR+MLIR operations that implement Fortran
/// generic intrinsic function calls. It operates purely on FIR+MLIR types so
/// that it can be used at different lowering level if needed.
/// IntrinsicLibrary is not in charge of generating code for the argument
/// expressions/symbols. These must be generated before and the resulting
/// mlir::Values are inputs for the IntrinsicLibrary operation generation.
///
/// The operations generated can be as simple as a single runtime library call
/// or they may fully implement the intrinsic without runtime help. This
/// depends on the IntrinsicLibrary::Implementation.
///
/// IntrinsicLibrary should not be assumed cheap to build since they may need
/// to build a representation of the target runtime before they can be used.
/// Once built, they are stateless and cannot be modified.
///

class IntrinsicLibrary {
public:
  /// Available runtime library versions.
  enum class Version { PgmathFast, PgmathRelaxed, PgmathPrecise, LLVM };

  /// Create an IntrinsicLibrary targeting the desired runtime library version.
  IntrinsicLibrary(Version, mlir::MLIRContext &);
  ~IntrinsicLibrary();
  /// Generate the FIR+MLIR operations for the generic intrinsic "name".
  /// On failure, returns a nullptr, else the returned mlir::Value is
  /// the returned Fortran intrinsic value.
  mlir::Value genval(mlir::Location loc, Fortran::lower::FirOpBuilder &builder,
                     llvm::StringRef name, mlir::Type resultType,
                     llvm::ArrayRef<mlir::Value> args) const;

  // TODO: Expose interface to get specific intrinsic function address.
  // TODO: Handle intrinsic subroutine.
  // TODO: Intrinsics that do not require their arguments to be defined
  //   (e.g shape inquiries) might not fit in the current interface that
  //   requires mlir::Value to be provided.
  // TODO: Error handling interface ?
  // TODO: Implementation is incomplete. Many intrinsics to tbd.

private:
  /// Actual implementation is hidden.
  class Implementation;
  std::unique_ptr<Implementation> impl;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_INTRINSICS_H_
