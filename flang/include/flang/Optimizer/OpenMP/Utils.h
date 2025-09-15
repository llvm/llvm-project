//===-- Optimizer/OpenMP/Utils.h --------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_OPENMP_UTILS_H
#define FORTRAN_OPTIMIZER_OPENMP_UTILS_H

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/DirectivesCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIRType.h"

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

namespace flangomp {

enum class DoConcurrentMappingKind {
  DCMK_None,  ///< Do not lower `do concurrent` to OpenMP.
  DCMK_Host,  ///< Lower to run in parallel on the CPU.
  DCMK_Device ///< Lower to run in parallel on the GPU.
};

/// Return true if the variable has a dynamic size and therefore requires
/// bounds operations to describe its extents.
inline bool needsBoundsOps(mlir::Value var) {
  assert(mlir::isa<mlir::omp::PointerLikeType>(var.getType()) &&
         "needsBoundsOps can deal only with pointer types");
  mlir::Type t = fir::unwrapRefType(var.getType());
  if (mlir::Type inner = fir::dyn_cast_ptrOrBoxEleTy(t))
    return fir::hasDynamicSize(inner);
  return fir::hasDynamicSize(t);
}

/// Generate MapBoundsOp operations for the variable and append them to
/// `boundsOps`.
inline llvm::SmallVector<mlir::Value> genBoundsOps(fir::FirOpBuilder &builder,
                                                   mlir::Value var,
                                                   bool isAssumedSize = false,
                                                   bool isOptional = false) {
  mlir::Location loc = var.getLoc();
  fir::factory::AddrAndBoundsInfo info =
      fir::factory::getDataOperandBaseAddr(builder, var, isOptional, loc);
  fir::ExtendedValue exv =
      hlfir::translateToExtendedValue(loc, builder, hlfir::Entity{info.addr},
                                      /*contiguousHint=*/true)
          .first;
  return fir::factory::genImplicitBoundsOps<mlir::omp::MapBoundsOp,
                                            mlir::omp::MapBoundsType>(
      builder, info, exv, isAssumedSize, loc);
}

} // namespace flangomp

#endif // FORTRAN_OPTIMIZER_OPENMP_UTILS_H
