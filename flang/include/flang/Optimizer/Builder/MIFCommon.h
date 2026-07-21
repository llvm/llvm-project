//===-- MIFCommon.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_
#define FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"

static constexpr llvm::StringRef coarrayHandleSuffix = "_coarray_handle";
static constexpr llvm::StringRef mifSaveCoarraysAllocName =
    "__mif_save_coarrays_allocate";

namespace mif {

std::string getFullUniqName(mlir::Value addr);

mlir::Value genImageIndex(fir::FirOpBuilder &, mlir::Location loc,
                          mlir::Value coarray, mlir::Value sub,
                          mlir::Value team);

mlir::func::FuncOp getOrCreateInitFunc(fir::FirOpBuilder &builder,
                                       mlir::ModuleOp mod,
                                       llvm::StringRef name);

} // namespace mif

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_MIFCOMMON_H_
