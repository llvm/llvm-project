//===-- Optimizer/Support/TypeCode.h ----------------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H
#define FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H

#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "mlir/IR/Types.h"

namespace fir {
/// Return the ISO_Fortran_binding.h type code for mlir type \p ty.
int getTypeCode(mlir::Type ty, KindMapping &kindMap);
} // namespace fir

#endif // FORTRAN_OPTIMIZER_SUPPORT_TYPECODE_H
