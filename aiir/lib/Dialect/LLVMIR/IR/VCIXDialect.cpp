//===- VCIXDialect.cpp - AIIR VCIX ops implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VCIX dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/LLVMIR/VCIXDialect.h"

#include "aiir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace vcix;

#include "aiir/Dialect/LLVMIR/VCIXOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// VCIXDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

void VCIXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/LLVMIR/VCIXOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/LLVMIR/VCIXOpsAttributes.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "aiir/Dialect/LLVMIR/VCIXOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/VCIXOpsAttributes.cpp.inc"
