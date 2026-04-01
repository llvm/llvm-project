//===-- MyExtension.h - Transform dialect tutorial --------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 4 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IR/TransformOps.h"
#include "aiir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace aiir {
class CallOpInterface;
namespace func {
class CallOp;
} // namespace func
} // namespace aiir

#define GET_OP_CLASSES
#include "MyExtension.h.inc"

// Registers our Transform dialect extension.
void registerMyExtension(::aiir::DialectRegistry &registry);
