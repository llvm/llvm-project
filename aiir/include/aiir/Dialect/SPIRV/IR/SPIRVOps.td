//===-- SPIRVOps.td - AIIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main operation definition specification file for SPIR-V
// operations.
//
//===----------------------------------------------------------------------===//

// Note that for each op in this file and the included files for specific op
// categories, we use a tool to automatically generate certain sections in its
// definition: basic structure, summary, description. So modifications to these
// sections will not be respected. Modifications to op traits, arguments,
// results, and sections after the results are retained. Besides, ops must be
// separated via the '// -----' marker.

#ifndef AIIR_DIALECT_SPIRV_IR_OPS
#define AIIR_DIALECT_SPIRV_IR_OPS

include "aiir/Dialect/SPIRV/IR/SPIRVBase.td"
include "aiir/Dialect/SPIRV/IR/SPIRVArithmeticOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVAtomicOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVBarrierOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVBitOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVCastOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVCompositeOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVControlFlowOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVCooperativeMatrixOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVIntelExtOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVGLOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVGraphOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVGroupOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVImageOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVIntegerDotProductOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVLogicalOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVMatrixOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVMemoryOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVMeshOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVMiscOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVNonUniformOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVPrimitiveOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVCLOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVStructureOps.td"
include "aiir/Dialect/SPIRV/IR/SPIRVTosaOps.td"
include "aiir/Interfaces/SideEffectInterfaces.td"

#endif // AIIR_DIALECT_SPIRV_IR_OPS
