//===-- OpenMPClauseOperands.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the structures defining AIIR operands associated with each
// OpenMP clause, and structures grouping the appropriate operands for each
// construct.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
#define AIIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_

#include "aiir/Dialect/OpenMP/OpenMPOpsAttributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "aiir/Dialect/OpenMP/OpenMPClauseOps.h.inc"

namespace aiir {
namespace omp {

//===----------------------------------------------------------------------===//
// Extra clause operand structures.
//===----------------------------------------------------------------------===//

struct DeviceTypeClauseOps {
  /// The default capture type.
  DeclareTargetDeviceType deviceType = DeclareTargetDeviceType::any;
};

//===----------------------------------------------------------------------===//
// Extra operation operand structures.
//===----------------------------------------------------------------------===//

/// Clauses that correspond to operations other than omp.target, but might have
/// to be evaluated outside of a parent target region.
using HostEvaluatedOperands =
    detail::Clauses<CollapseClauseOps, LoopRelatedClauseOps, NumTeamsClauseOps,
                    NumThreadsClauseOps, ThreadLimitClauseOps>;

// TODO: Add `indirect` clause.
using DeclareTargetOperands = detail::Clauses<DeviceTypeClauseOps>;

/// omp.target_enter_data, omp.target_exit_data and omp.target_update take the
/// same clauses, so we give the structure to be shared by all of them a
/// representative name.
using TargetEnterExitUpdateDataOperands = TargetEnterDataOperands;

} // namespace omp
} // namespace aiir

#endif // AIIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
