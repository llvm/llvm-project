//===- VCIXDialect.h - AIIR VCIX IR dialect -------------------*- C++ ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file defines the basic operations for the VCIX dialect.
//
// The SiFive Vector Coprocessor Interface (VCIX) provides a flexible mechanism
// to extend application processors with custom coprocessors and
// variable-latency arithmetic units. The interface offers throughput comparable
// to that of standard RISC-V vector instructions. To accelerate performance,
// system designers may use VCIX as a low-latency, high-throughput interface to
// a coprocessor
//
// https://www.sifive.com/document-file/sifive-vector-coprocessor-interface-vcix-software
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_VCIXDIALECT_H_
#define AIIR_DIALECT_LLVMIR_VCIXDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/VCIXOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/LLVMIR/VCIXOps.h.inc"

#include "aiir/Dialect/LLVMIR/VCIXOpsDialect.h.inc"

#endif /* AIIR_DIALECT_LLVMIR_VCIXDIALECT_H_ */
