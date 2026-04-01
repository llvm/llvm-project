//===- BytecodeOpInterface.h - Bytecode interface for AIIR Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the BytecodeOpInterface defined in
// `BytecodeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_BYTECODE_BYTECODEOPINTERFACE_H
#define AIIR_BYTECODE_BYTECODEOPINTERFACE_H

#include "aiir/Bytecode/BytecodeImplementation.h"
#include "aiir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "aiir/Bytecode/BytecodeOpInterface.h.inc"

#endif // AIIR_BYTECODE_BYTECODEOPINTERFACE_H
