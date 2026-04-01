//===- ROCDLDialect.h - AIIR ROCDL IR dialect -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ROCDL dialect in AIIR, containing ROCDL operations
// and ROCDL specific extensions to the LLVM type system.
//
// Unfortunately there does not exists a formal definition of ROCDL IR that be
// pointed to here. However the following links contain more information about
// ROCDL (ROCm-Device-Library)
//
// https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/doc/OCML.md
// https://github.com/ROCm/llvm-project/blob/amd-staging/amd/device-libs/doc/OCKL.md
// https://llvm.org/docs/AMDGPUUsage.html
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_ROCDLDIALECT_H_
#define AIIR_DIALECT_LLVMIR_ROCDLDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/ROCDLOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/LLVMIR/ROCDLOps.h.inc"

#include "aiir/Dialect/LLVMIR/ROCDLOpsDialect.h.inc"

#endif /* AIIR_DIALECT_LLVMIR_ROCDLDIALECT_H_ */
