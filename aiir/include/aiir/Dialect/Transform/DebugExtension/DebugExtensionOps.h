//===- DebugExtensionOps.h - Debug ext. for Transform dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DEBUGEXTENSIONOPS_H
#define AIIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DEBUGEXTENSIONOPS_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "aiir/Dialect/Transform/DebugExtension/DebugExtensionOps.h.inc"

#endif // AIIR_DIALECT_TRANSFORM_DEBUGEXTENSION_DEBUGEXTENSIONOPS_H
