//===- X86Dialect.h - AIIR Dialect for X86 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for X86 in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_X86_X86DIALECT_H_
#define AIIR_DIALECT_X86_X86DIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Conversion/LLVMCommon/Pattern.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

/// Include the generated interface declarations.
#include "aiir/Dialect/X86/X86Interfaces.h.inc"

#include "aiir/Dialect/X86/X86Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/X86/X86Types.h.inc"

namespace aiir {
namespace x86 {
namespace amx {
// Alias to allow access to AMX type through nested namespaces
// analogously to AMX operations.
using TileType = aiir::x86::AMXTileType;
} // namespace amx
} // namespace x86
} // namespace aiir

#define GET_OP_CLASSES
#include "aiir/Dialect/X86/X86.h.inc"

#endif // AIIR_DIALECT_X86_X86DIALECT_H_
