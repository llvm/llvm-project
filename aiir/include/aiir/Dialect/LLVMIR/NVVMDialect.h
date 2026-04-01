//===- NVVMDialect.h - AIIR NVVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NVVM IR dialect in AIIR, containing NVVM operations and
// NVVM specific extensions to the LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_LLVMIR_NVVMDIALECT_H_
#define AIIR_DIALECT_LLVMIR_NVVMDIALECT_H_

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"
#include "aiir/Dialect/LLVMIR/BasicPtxBuilderInterface.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/NVVMRequiresSMTraits.h"
#include "aiir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/InferIntRangeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

#include "aiir/Dialect/LLVMIR/NVVMOpsEnums.h.inc"

namespace aiir {
namespace NVVM {
/// Utility functions to compare NVVMMemorySpace with unsigned values.
inline bool operator==(unsigned as, NVVMMemorySpace memSpace) {
  return as == static_cast<unsigned>(memSpace);
}
inline bool operator==(NVVMMemorySpace memSpace, unsigned as) {
  return static_cast<unsigned>(memSpace) == as;
}
inline bool operator!=(unsigned as, NVVMMemorySpace memSpace) {
  return as != static_cast<unsigned>(memSpace);
}
inline bool operator!=(NVVMMemorySpace memSpace, unsigned as) {
  return static_cast<unsigned>(memSpace) != as;
}

// Shared memory has 128-bit alignment
constexpr int kSharedMemoryAlignmentBit = 128;

/// A pair type of LLVM's Intrinsic ID and args (which are llvm values).
/// This type is returned by the getIntrinsicIDAndArgs() methods.
using IDArgPair =
    std::pair<llvm::Intrinsic::ID, llvm::SmallVector<llvm::Value *>>;

/// Return the element type and number of elements associated with a wmma matrix
/// of given chracteristics. This matches the logic in IntrinsicsNVVM.td
/// WMMA_REGS structure.
std::pair<aiir::Type, unsigned> inferMMAType(aiir::NVVM::MMATypes type,
                                             aiir::NVVM::MMAFrag frag, int nRow,
                                             int nCol,
                                             aiir::AIIRContext *context);
} // namespace NVVM
} // namespace aiir

///// Ops /////
#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/LLVMIR/NVVMOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/LLVMIR/NVVMOps.h.inc"

#include "aiir/Dialect/LLVMIR/NVVMOpsDialect.h.inc"

#endif /* AIIR_DIALECT_LLVMIR_NVVMDIALECT_H_ */
