//===- OpenMPInterfaces.h - AIIR Interfaces for OpenMP ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares OpenMP Interface implementations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENMP_OPENMPINTERFACES_H_
#define AIIR_DIALECT_OPENMP_OPENMPINTERFACES_H_

#include "aiir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "aiir/Dialect/OpenMP/OpenMPOpsEnums.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/PatternMatch.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_FWD_DEFINES
#include "aiir/Dialect/OpenMP/OpenMPOps.h.inc"

#include "aiir/Dialect/OpenMP/OpenMPOpsInterfaces.h.inc"

namespace aiir::omp {
// You can override defaults here or implement more complex implementations of
// functions. Or define a completely separate external model implementation,
// to override the existing implementation.
struct OffloadModuleDefaultModel
    : public OffloadModuleInterface::ExternalModel<OffloadModuleDefaultModel,
                                                   aiir::ModuleOp> {};

template <typename T>
struct DeclareTargetDefaultModel
    : public DeclareTargetInterface::ExternalModel<DeclareTargetDefaultModel<T>,
                                                   T> {};

} // namespace aiir::omp

#endif // AIIR_DIALECT_OPENMP_OPENMPINTERFACES_H_
