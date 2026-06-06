//===-- include/flang/Support/OpenMP-utils.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SUPPORT_OPENMP_UTILS_H_
#define FORTRAN_SUPPORT_OPENMP_UTILS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace Fortran::common::openmp {
/// Structure holding the information needed to create and bind entry block
/// arguments associated to all clauses that can define them.
struct EntryBlockArgs {
  llvm::ArrayRef<mlir::Value> hasDeviceAddrVars;
  llvm::ArrayRef<mlir::Value> hostEvalVars;
  llvm::ArrayRef<mlir::Value> inReductionVars;
  llvm::ArrayRef<mlir::Value> mapVars;
  llvm::ArrayRef<mlir::Value> privVars;
  llvm::ArrayRef<mlir::Value> reductionVars;
  llvm::ArrayRef<mlir::Value> taskReductionVars;
  llvm::ArrayRef<mlir::Value> useDeviceAddrVars;
  llvm::ArrayRef<mlir::Value> useDevicePtrVars;

  auto getVars() const {
    return llvm::concat<const mlir::Value>(hasDeviceAddrVars, hostEvalVars,
        inReductionVars, mapVars, privVars, reductionVars, taskReductionVars,
        useDeviceAddrVars, useDevicePtrVars);
  }
};

/// Create an entry block for the given region, including the clause-defined
/// arguments specified.
///
/// \param [in] builder - MLIR operation builder.
/// \param [in]    args - entry block arguments information for the given
///                       operation.
/// \param [in]  region - Empty region in which to create the entry block.
mlir::Block *genEntryBlock(
    mlir::OpBuilder &builder, const EntryBlockArgs &args, mlir::Region &region);
} // namespace Fortran::common::openmp

#endif // FORTRAN_SUPPORT_OPENMP_UTILS_H_
