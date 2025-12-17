//===-- Lower/OpenMP/PrivateReductionUtils.h --------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_OPENMP_PRIVATEREDUCTIONUTILS_H
#define FORTRAN_LOWER_OPENMP_PRIVATEREDUCTIONUTILS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace mlir {
class Region;
} // namespace mlir

namespace Fortran {
namespace semantics {
class Symbol;
} // namespace semantics
} // namespace Fortran

namespace fir {
class FirOpBuilder;
class ShapeShiftOp;
} // namespace fir

namespace Fortran {
namespace lower {
class AbstractConverter;

enum class DeclOperationKind {
  PrivateOrLocal,
  FirstPrivateOrLocalInit,
  Reduction
};
inline bool isPrivatization(DeclOperationKind kind) {
  return (kind == DeclOperationKind::FirstPrivateOrLocalInit) ||
         (kind == DeclOperationKind::PrivateOrLocal);
}
inline bool isReduction(DeclOperationKind kind) {
  return kind == DeclOperationKind::Reduction;
}

/// Generate init and cleanup regions suitable for reduction or privatizer
/// declarations. `scalarInitValue` may be nullptr if there is no default
/// initialization (for privatization). `kind` should be set to indicate
/// what kind of operation definition this initialization belongs to.
void populateByRefInitAndCleanupRegions(
    AbstractConverter &converter, mlir::Location loc, mlir::Type argType,
    mlir::Value scalarInitValue, mlir::Block *initBlock,
    mlir::Value allocatedPrivVarArg, mlir::Value moldArg,
    mlir::Region &cleanupRegion, DeclOperationKind kind,
    const Fortran::semantics::Symbol *sym = nullptr,
    bool cannotHaveNonDefaultLowerBounds = false, bool isDoConcurrent = false);

/// Generate a fir::ShapeShift op describing the provided boxed array.
/// `cannotHaveNonDefaultLowerBounds` should be set if `box` is known to have
/// default lower bounds. This can improve code generation.
/// `useDefaultLowerBounds` can be set to force the returned fir::ShapeShiftOp
/// to have default lower bounds, which is useful to iterate through array
/// elements without having to adjust each index.
fir::ShapeShiftOp getShapeShift(fir::FirOpBuilder &builder, mlir::Location loc,
                                mlir::Value box,
                                bool cannotHaveNonDefaultLowerBounds = false,
                                bool useDefaultLowerBounds = false);

} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_OPENMP_PRIVATEREDUCTIONUTILS_H
