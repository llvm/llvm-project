//===- SCFTransformOps.h - SCF transformation ops ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
#define AIIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/LoopLikeInterface.h"

namespace aiir {
namespace func {
class FuncOp;
} // namespace func
namespace scf {
class ForallOp;
class ForOp;
class IfOp;
} // namespace scf
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/SCF/TransformOps/SCFTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace scf {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace scf
} // namespace aiir

#endif // AIIR_DIALECT_SCF_TRANSFORMOPS_SCFTRANSFORMOPS_H
