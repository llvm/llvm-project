//===-- Optimizer/Dialect/CUF/CUFOps.h - CUF operations ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_DIALECT_CUF_CUFOPS_H
#define FORTRAN_OPTIMIZER_DIALECT_CUF_CUFOPS_H

#include "flang/Optimizer/Dialect/CUF/Attributes/CUFAttr.h"
#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "flang/Optimizer/Dialect/CUF/CUFOps.h.inc"

#endif // FORTRAN_OPTIMIZER_DIALECT_CUF_CUFOPS_H
