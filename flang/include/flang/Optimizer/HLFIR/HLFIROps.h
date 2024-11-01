//===-- HLFIROps.h - FIR operations -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_HLFIR_HLFIROPS_H
#define FORTRAN_OPTIMIZER_HLFIR_HLFIROPS_H

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FortranVariableInterface.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "flang/Optimizer/HLFIR/HLFIROps.h.inc"

#endif // FORTRAN_OPTIMIZER_HLFIR_HLFIROPS_H
