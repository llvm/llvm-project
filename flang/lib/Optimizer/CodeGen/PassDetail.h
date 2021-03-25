//===- PassDetail.h - Optimizer code gen Pass class details -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTMIZER_CODEGEN_PASSDETAIL_H
#define OPTMIZER_CODEGEN_PASSDETAIL_H

#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace fir {

#define GEN_PASS_CLASSES
#include "flang/Optimizer/CodeGen/CGPasses.h.inc"

} // namespace fir

#endif // OPTMIZER_CODEGEN_PASSDETAIL_H
