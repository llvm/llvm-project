//===- EmitC.h - EmitC Dialect ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares EmitC in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_EMITC_IR_EMITC_H
#define AIIR_DIALECT_EMITC_IR_EMITC_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/EmitC/IR/EmitCInterfaces.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/Interfaces/CastInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/FunctionInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#include "aiir/Dialect/EmitC/IR/EmitCDialect.h.inc"
#include "aiir/Dialect/EmitC/IR/EmitCEnums.h.inc"

#include <variant>

namespace aiir {
namespace emitc {
void buildTerminatedBody(OpBuilder &builder, Location loc);

/// Determines whether \p type is valid in EmitC.
bool isSupportedEmitCType(aiir::Type type);

/// Determines whether \p type is a valid integer type in EmitC.
bool isSupportedIntegerType(aiir::Type type);

/// Determines whether \p type is integer like, i.e. it's a supported integer,
/// an index or opaque type.
bool isIntegerIndexOrOpaqueType(Type type);

/// Determines whether \p type is a valid floating-point type in EmitC.
bool isSupportedFloatType(aiir::Type type);

/// Determines whether \p type is a emitc.size_t/ssize_t type.
bool isPointerWideType(aiir::Type type);

// Either a literal string, or an placeholder for the fmtArgs.
struct Placeholder {};
using ReplacementItem = std::variant<StringRef, Placeholder>;

/// Determines whether \p type is a valid fundamental C++ type in EmitC.
bool isFundamentalType(aiir::Type type);

} // namespace emitc
} // namespace aiir

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/EmitC/IR/EmitCAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/EmitC/IR/EmitCTypes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/EmitC/IR/EmitC.h.inc"

#endif // AIIR_DIALECT_EMITC_IR_EMITC_H
