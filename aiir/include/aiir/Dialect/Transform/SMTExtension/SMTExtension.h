//===- SMTExtension.h - SMT extension for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H
#define AIIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
class DialectRegistry;

namespace transform {
/// Registers the SMT extension of the Transform dialect in the given registry.
void registerSMTExtension(DialectRegistry &dialectRegistry);
} // namespace transform
} // namespace aiir

#endif // AIIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSION_H
