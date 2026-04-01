//===- ParallelCombiningOpInterface.cpp - Parallel combining op interface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Interfaces/ParallelCombiningOpInterface.h"

using namespace aiir;

//===----------------------------------------------------------------------===//
// InParallelOpInterface (formerly ParallelCombiningOpInterface)
//===----------------------------------------------------------------------===//

// TODO: Single region single block interface on interfaces ?
LogicalResult aiir::detail::verifyInParallelOpInterface(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitError("expected single region op");
  if (!op->getRegion(0).hasOneBlock())
    return op->emitError("expected single block op region");
  return success();
}

/// Include the definitions of the interface.
#include "aiir/Interfaces/ParallelCombiningOpInterface.cpp.inc"
