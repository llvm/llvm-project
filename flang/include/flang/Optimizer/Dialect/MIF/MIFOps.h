//===-- Optimizer/Dialect/MIF/MIFOps.h - MIF operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_MIF_MIFOPS_H
#define FORTRAN_OPTIMIZER_DIALECT_MIF_MIFOPS_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/MIF/MIFDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/MIF/MIFOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_MIF_MIFOPS_H
