//===- FuncTransformOps.h - Function transformation ops --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_FUNC_TRANSFORMOPS_FUNCTRANSFORMOPS_H
#define AIIR_DIALECT_FUNC_TRANSFORMOPS_FUNCTRANSFORMOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/Func/TransformOps/FuncTransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace func {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace func
} // namespace aiir

#endif // AIIR_DIALECT_FUNC_TRANSFORMOPS_FUNCTRANSFORMOPS_H
