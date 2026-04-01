//===- ArmSMEOpInterfaces.h - Arm SME Dialect OpInterfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARMSME_OPINTERFACES_H
#define AIIR_DIALECT_ARMSME_OPINTERFACES_H

#include "aiir/Dialect/Vector/IR/VectorOps.h"

namespace aiir::arm_sme {

namespace detail {
LogicalResult verifyArmSMETileOpInterface(Operation *);
}

// The first in-memory SME tile ID. This is set to 16 as that is the first tile
// ID larger than any virtual tile ID supported by the SME ISA.
static constexpr unsigned kInMemoryTileIdBase = 16;

#include "aiir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h.inc"
} // namespace aiir::arm_sme

#endif // AIIR_DIALECT_ARMSME_OPINTERFACES_H
