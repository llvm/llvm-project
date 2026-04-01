//===- IR.h - C API Utils for Core AIIR classes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains declarations of implementation details of the C API for
// core AIIR classes. This file should not be included from C++ code other than
// C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CAPI_IR_H
#define AIIR_CAPI_IR_H

#include "aiir/Bytecode/BytecodeWriter.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Operation.h"

DEFINE_C_API_PTR_METHODS(AiirAsmState, aiir::AsmState)
DEFINE_C_API_PTR_METHODS(AiirBytecodeWriterConfig, aiir::BytecodeWriterConfig)
DEFINE_C_API_PTR_METHODS(AiirContext, aiir::AIIRContext)
DEFINE_C_API_PTR_METHODS(AiirDialect, aiir::Dialect)
DEFINE_C_API_PTR_METHODS(AiirDialectRegistry, aiir::DialectRegistry)
DEFINE_C_API_PTR_METHODS(AiirOperation, aiir::Operation)
DEFINE_C_API_PTR_METHODS(AiirBlock, aiir::Block)
DEFINE_C_API_PTR_METHODS(AiirOpOperand, aiir::OpOperand)
DEFINE_C_API_PTR_METHODS(AiirOpPrintingFlags, aiir::OpPrintingFlags)
DEFINE_C_API_PTR_METHODS(AiirRegion, aiir::Region)
DEFINE_C_API_PTR_METHODS(AiirSymbolTable, aiir::SymbolTable)

DEFINE_C_API_METHODS(AiirAttribute, aiir::Attribute)
DEFINE_C_API_METHODS(AiirIdentifier, aiir::StringAttr)
DEFINE_C_API_METHODS(AiirLocation, aiir::Location)
DEFINE_C_API_METHODS(AiirModule, aiir::ModuleOp)
DEFINE_C_API_METHODS(AiirType, aiir::Type)
DEFINE_C_API_METHODS(AiirValue, aiir::Value)

#endif // AIIR_CAPI_IR_H
