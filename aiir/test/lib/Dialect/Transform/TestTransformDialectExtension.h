//===- TestTransformDialectExtension.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the AIIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTTRANSFORMDIALECTEXTENSION_H
#define AIIR_TESTTRANSFORMDIALECTEXTENSION_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/PDL/IR/PDLTypes.h"
#include "aiir/Dialect/Transform/IR/TransformTypes.h"
#include "aiir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpImplementation.h"

namespace aiir {
class DialectRegistry;
} // namespace aiir

#define GET_TYPEDEF_CLASSES
#include "TestTransformDialectExtensionTypes.h.inc"

#define GET_OP_CLASSES
#include "TestTransformDialectExtension.h.inc"

namespace test {
/// Registers the test extension to the Transform dialect.
void registerTestTransformDialectExtension(::aiir::DialectRegistry &registry);
} // namespace test

#endif // AIIR_TESTTRANSFORMDIALECTEXTENSION_H
