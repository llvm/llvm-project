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

namespace flangomp {

enum class DoConcurrentMappingKind {
  DCMK_None,  ///< Do not lower `do concurrent` to OpenMP.
  DCMK_Host,  ///< Lower to run in parallel on the CPU.
  DCMK_Device ///< Lower to run in parallel on the GPU.
};

} // namespace flangomp

#endif // FORTRAN_OPTIMIZER_OPENMP_UTILS_H
