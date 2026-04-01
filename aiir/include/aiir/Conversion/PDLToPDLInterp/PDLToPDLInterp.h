//===- PDLToPDLInterp.h - PDL to PDL Interpreter conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass for PDL to PDL Interpreter dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H
#define AIIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H

#include "aiir/Pass/Pass.h"
#include "aiir/Support/LLVM.h"

namespace aiir {
class ModuleOp;
class Operation;
template <typename OpT>
class OperationPass;
class PDLPatternConfigSet;

#define GEN_PASS_DECL_CONVERTPDLTOPDLINTERPPASS
#include "aiir/Conversion/Passes.h.inc"

/// Creates and returns a pass to convert PDL ops to PDL interpreter ops.
/// `configMap` holds a map of the configurations for each pattern being
/// compiled.
std::unique_ptr<OperationPass<ModuleOp>> createConvertPDLToPDLInterpPass(
    DenseMap<Operation *, PDLPatternConfigSet *> &configMap);

} // namespace aiir

#endif // AIIR_CONVERSION_PDLTOPDLINTERP_PDLTOPDLINTERP_H
