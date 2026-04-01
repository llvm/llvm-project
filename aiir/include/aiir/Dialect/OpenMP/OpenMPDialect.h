//===- OpenMPDialect.h - AIIR Dialect for OpenMP ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OpenMP dialect in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENMP_OPENMPDIALECT_H_
#define AIIR_DIALECT_OPENMP_OPENMPDIALECT_H_

#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "aiir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.h"
#include "aiir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "aiir/Dialect/OpenMP/OpenMPOffloadUtils.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/OpenMP/OpenMPOpsTypes.h.inc"

#include "aiir/Dialect/OpenMP/OpenMPOpsDialect.h.inc"

#include "aiir/Dialect/OpenMP/OpenMPClauseOperands.h"

#include "aiir/Dialect/OpenMP/OpenMPTypeInterfaces.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/OpenMP/OpenMPOps.h.inc"

/// Operations implementing LoopWrapperInterface.
#define OMP_LOOP_WRAPPER_OPS                                                   \
  aiir::omp::WorkshareLoopWrapperOp, aiir::omp::LoopOp, aiir::omp::WsloopOp,   \
      aiir::omp::SimdOp, aiir::omp::DistributeOp, aiir::omp::TaskloopOp

/// Operations implementing OutlineableOpenMPOpInterface.
#define OMP_OUTLINEABLE_OPS                                                    \
  aiir::omp::ParallelOp, aiir::omp::TeamsOp, aiir::omp::TaskOp,                \
      aiir::omp::TargetOp

namespace aiir::omp {
/// Find the omp.new_cli, generator, and consumer of a canonical loop info.
std::tuple<NewCliOp, OpOperand *, OpOperand *> decodeCli(aiir::Value cli);
} // namespace aiir::omp

#endif // AIIR_DIALECT_OPENMP_OPENMPDIALECT_H_
