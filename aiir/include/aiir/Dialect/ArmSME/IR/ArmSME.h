//===- ArmSMEDialect.h - AIIR Dialect for Arm SME ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for ArmSME in AIIR.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARMSME_IR_ARMSME_H
#define AIIR_DIALECT_ARMSME_IR_ARMSME_H

#include "aiir/Bytecode/BytecodeOpInterface.h"
#include "aiir/Dialect/ArmSME/IR/ArmSMEEnums.h"
#include "aiir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h"
#include "aiir/Dialect/ArmSME/Utils/Utils.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "aiir/Dialect/SCF/IR/SCF.h"
#include "aiir/Dialect/Vector/IR/VectorOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEAttrDefs.h.inc"

#include "aiir/Dialect/ArmSME/IR/ArmSMEDialect.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEOps.h.inc"

#define GET_OP_CLASSES
#include "aiir/Dialect/ArmSME/IR/ArmSMEIntrinsicOps.h.inc"

#endif // AIIR_DIALECT_ARMSME_IR_ARMSME_H
