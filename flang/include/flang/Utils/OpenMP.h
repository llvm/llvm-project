//===-- include/flang/Utils/OpenMP.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_UTILS_OPENMP_H_
#define FORTRAN_UTILS_OPENMP_H_

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace fir {
class FirOpBuilder;
} // namespace fir

namespace Fortran::utils::openmp {
// TODO We can probably move the stuff inside `Support/OpenMP-utils.h/.cpp` here
// as well.

/// Create an `omp.map.info` op. Parameters other than the ones documented below
/// correspond to operation arguments in the OpenMPOps.td file, see op docs for
/// more details.
///
/// \param [in] builder - MLIR operation builder.
/// \param [in] loc     - Source location of the created op.
mlir::omp::MapInfoOp createMapInfoOp(mlir::OpBuilder &builder,
    mlir::Location loc, mlir::Value baseAddr, mlir::Value varPtrPtr,
    llvm::StringRef name, llvm::ArrayRef<mlir::Value> bounds,
    llvm::ArrayRef<mlir::Value> members, mlir::ArrayAttr membersIndex,
    uint64_t mapType, mlir::omp::VariableCaptureKind mapCaptureType,
    mlir::Type retTy, bool partialMap = false,
    mlir::FlatSymbolRefAttr mapperId = mlir::FlatSymbolRefAttr());

/// For an mlir value that does not have storage, allocate temporary storage
/// (outside the target region), store the value in that storage, and map the
/// storage to the target region.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - Target op to which the temporary value is mapped.
/// \param val          - Temp value that should be mapped to the target region.
/// \param name         - A string used to identify the created `omp.map.info`
/// op.
///
/// \returns The loaded mapped value inside the target region.
mlir::Value mapTemporaryValue(fir::FirOpBuilder &firOpBuilder,
    mlir::omp::TargetOp targetOp, mlir::Value val,
    llvm::StringRef name = "tmp.map");

/// For values used inside a target region but defined outside, either clone
/// these value inside the target region or map them to the region. This
/// function first tries to clone values (if they are defined by
/// memory-effect-free ops, otherwise, the values are mapped.
///
/// \param firOpBuilder - Operation builder.
/// \param targetOp     - The target that needs to be extended by clones and/or
/// maps.
void cloneOrMapRegionOutsiders(
    fir::FirOpBuilder &firOpBuilder, mlir::omp::TargetOp targetOp);
} // namespace Fortran::utils::openmp

#endif // FORTRAN_UTILS_OPENMP_H_
