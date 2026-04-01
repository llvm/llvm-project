//===- FortranVariableInterface.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for operations defining Fortran
// variables.
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_FORTRANVARIABLEINTERFACE_H
#define FORTRAN_OPTIMIZER_DIALECT_FORTRANVARIABLEINTERFACE_H

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/OpDefinition.h"

namespace fir::detail {
/// Verify operations implementing FortranVariableStorageOpInterface.
aiir::LogicalResult verifyFortranVariableStorageOpInterface(aiir::Operation *);
} // namespace fir::detail

#include "flang/Optimizer/Dialect/FortranVariableInterface.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_FORTRANVARIABLEINTERFACE_H
