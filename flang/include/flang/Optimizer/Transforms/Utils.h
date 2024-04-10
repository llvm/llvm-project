//===-- Optimizer/Transforms/Utils.h ----------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H

namespace fir {

using MinlocBodyOpGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value, mlir::Value, const llvm::SmallVectorImpl<mlir::Value> &)>;
using InitValGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &)>;
using AddrGeneratorTy = llvm::function_ref<mlir::Value(
    fir::FirOpBuilder &, mlir::Location, const mlir::Type &, mlir::Value,
    mlir::Value)>;

// Produces a loop nest for a Minloc intrinsic.
void genMinMaxlocReductionLoop(fir::FirOpBuilder &builder, mlir::Value array,
                               fir::InitValGeneratorTy initVal,
                               fir::MinlocBodyOpGeneratorTy genBody,
                               fir::AddrGeneratorTy getAddrFn, unsigned rank,
                               mlir::Type elementType, mlir::Location loc,
                               mlir::Type maskElemType, mlir::Value resultArr,
                               bool maskMayBeLogicalScalar);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H
