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
} // namespace Fortran::utils::openmp

#endif // FORTRAN_UTILS_OPENMP_H_
