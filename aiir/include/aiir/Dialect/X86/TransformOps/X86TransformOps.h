//===- X86TransformOps.h - X86 transform ops --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_X86_TRANSFORMOPS_X86TRANSFORMOPS_H
#define AIIR_DIALECT_X86_TRANSFORMOPS_X86TRANSFORMOPS_H

#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

//===----------------------------------------------------------------------===//
// X86 Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aiir/Dialect/X86/TransformOps/X86TransformOps.h.inc"

namespace aiir {
class DialectRegistry;

namespace x86 {
void registerTransformDialectExtension(DialectRegistry &registry);

} // namespace x86
} // namespace aiir

#endif // AIIR_DIALECT_X86_TRANSFORMOPS_X86TRANSFORMOPS_H
