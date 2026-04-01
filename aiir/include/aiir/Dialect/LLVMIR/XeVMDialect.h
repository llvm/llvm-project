//===-- XeVMDialect.h - AIIR XeVM target definitions ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_XEVMDIALECT_H_
#define AIIR_DIALECT_LLVMIR_XEVMDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"

#include "aiir/Dialect/LLVMIR/XeVMOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/XeVMOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/LLVMIR/XeVMOps.h.inc"

#include "aiir/Dialect/LLVMIR/XeVMOpsDialect.h.inc"

#endif /* AIIR_DIALECT_LLVMIR_XEVMDIALECT_H_ */
