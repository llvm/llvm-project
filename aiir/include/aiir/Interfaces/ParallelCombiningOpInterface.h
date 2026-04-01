//===- ParallelCombiningOpInterface.h - Parallel combining op interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for ops that parallel combining
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_
#define AIIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_

#include "aiir/IR/OpDefinition.h"

namespace aiir {
namespace detail {
// TODO: Single region single block interface on interfaces ?
LogicalResult verifyInParallelOpInterface(Operation *op);
} // namespace detail
} // namespace aiir

/// Include the generated interface declarations.
#include "aiir/Interfaces/ParallelCombiningOpInterface.h.inc"

#endif // AIIR_INTERFACES_PARALLELCOMBININGOPINTERFACE_H_
