//===- MLProgramDialect.cpp - MLProgram dialect implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MLProgram/IR/MLProgram.h"
#include "aiir/IR/DialectImplementation.h"
#include "aiir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace aiir;
using namespace aiir::ml_program;

//===----------------------------------------------------------------------===//
/// Tablegen Definitions
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/MLProgram/IR/MLProgramOpsDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "aiir/Dialect/MLProgram/IR/MLProgramAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "aiir/Dialect/MLProgram/IR/MLProgramTypes.cpp.inc"

namespace {

struct MLProgramInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const override {
    // We have no specific opinion on whether ops defined in this dialect should
    // be inlined.
    return true;
  }
};

struct MLProgramOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
};
} // namespace

void ml_program::MLProgramDialect::initialize() {
#define GET_ATTRDEF_LIST
  addAttributes<
#include "aiir/Dialect/MLProgram/IR/MLProgramAttributes.cpp.inc"
      >();

#define GET_TYPEDEF_LIST
  addTypes<
#include "aiir/Dialect/MLProgram/IR/MLProgramTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "aiir/Dialect/MLProgram/IR/MLProgramOps.cpp.inc"
      >();

  addInterfaces<MLProgramInlinerInterface, MLProgramOpAsmDialectInterface>();
}
