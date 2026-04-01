//===-- Optimizer/Transforms/Utils.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://aiir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H
#define FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H

namespace fir {

using MinlocBodyOpGeneratorTy = llvm::function_ref<aiir::Value(
    fir::FirOpBuilder &, aiir::Location, const aiir::Type &, aiir::Value,
    aiir::Value, aiir::Value, const llvm::SmallVectorImpl<aiir::Value> &)>;
using InitValGeneratorTy = llvm::function_ref<aiir::Value(
    fir::FirOpBuilder &, aiir::Location, const aiir::Type &)>;
using AddrGeneratorTy = llvm::function_ref<aiir::Value(
    fir::FirOpBuilder &, aiir::Location, const aiir::Type &, aiir::Value,
    aiir::Value)>;

// Produces a loop nest for a Minloc intrinsic.
void genMinMaxlocReductionLoop(fir::FirOpBuilder &builder, aiir::Value array,
                               fir::InitValGeneratorTy initVal,
                               fir::MinlocBodyOpGeneratorTy genBody,
                               fir::AddrGeneratorTy getAddrFn, unsigned rank,
                               aiir::Type elementType, aiir::Location loc,
                               aiir::Type maskElemType, aiir::Value resultArr,
                               bool maskMayBeLogicalScalar);

} // namespace fir

#endif // FORTRAN_OPTIMIZER_TRANSFORMS_UTILS_H
