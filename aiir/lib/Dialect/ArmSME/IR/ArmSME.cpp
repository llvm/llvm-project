//===- ArmSMEDialect.cpp - AIIR ArmSME dialect implementation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ArmSME dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmSME/IR/ArmSME.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::arm_sme;

namespace aiir::arm_sme::detail {
LogicalResult verifyArmSMETileOpInterface(Operation *op) {
  return verifyOperationHasValidTileId(op);
}
} // namespace aiir::arm_sme::detail

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ArmSME/IR/ArmSMEDialect.cpp.inc"

#include "aiir/Dialect/ArmSME/IR/ArmSMEEnums.cpp.inc"

#include "aiir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEOps.cpp.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMETypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEAttrDefs.cpp.inc"

void ArmSMEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aiir/Dialect/ArmSME/IR/ArmSMEAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/ArmSME/IR/ArmSMEOps.cpp.inc"
      ,
#define GET_OP_LIST
#include "aiir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.cpp.inc"
      >();
}
