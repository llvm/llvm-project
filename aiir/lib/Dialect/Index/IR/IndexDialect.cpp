//===- IndexDialect.cpp - Index dialect definition -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Index/IR/IndexDialect.h"
#include "aiir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "aiir/Transforms/InliningUtils.h"

using namespace aiir;
using namespace aiir::index;

//===----------------------------------------------------------------------===//
// IndexDialect
//===----------------------------------------------------------------------===//
namespace {
/// This class defines the interface for handling inlining for index
/// dialect operations.
struct IndexInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All index dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void IndexDialect::initialize() {
  registerAttributes();
  registerOperations();
  addInterfaces<IndexInlinerInterface>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, IndexDialect>();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Index/IR/IndexOpsDialect.cpp.inc"
