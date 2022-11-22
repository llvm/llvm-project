//===- HLFIRDialect.h - High Level Fortran IR dialect -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HLFIR dialect that models Fortran expressions and
// assignments without requiring storage allocation and manipulations.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
#define FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H

#include "mlir/IR/Dialect.h"

namespace hlfir {
/// Is this a type that can be used for an HLFIR variable ?
bool isFortranVariableType(mlir::Type);
} // namespace hlfir

#include "flang/Optimizer/HLFIR/HLFIRDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "flang/Optimizer/HLFIR/HLFIRAttributes.h.inc"

#endif // FORTRAN_OPTIMIZER_HLFIR_HLFIRDIALECT_H
