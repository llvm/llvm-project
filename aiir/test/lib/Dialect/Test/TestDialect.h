//===- TestDialect.h - AIIR Dialect for testing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a fake 'test' dialect that can be used for testing things
// that do not have a respective counterpart in the main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTDIALECT_H
#define AIIR_TESTDIALECT_H

#include "TestAttributes.h"
#include "TestInterfaces.h"
#include "TestTypes.h"
#include "aiir/Bytecode/BytecodeImplementation.h"
#include "aiir/Dialect/Bufferization/IR/Bufferization.h"
#include "aiir/Dialect/DLTI/DLTI.h"
#include "aiir/Dialect/DLTI/Traits.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Dialect/Traits.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/DialectResourceBlobManager.h"
#include "aiir/IR/ExtensibleDialect.h"
#include "aiir/IR/OpDefinition.h"
#include "aiir/IR/OpImplementation.h"
#include "aiir/IR/RegionKindInterface.h"
#include "aiir/IR/SymbolTable.h"
#include "aiir/Interfaces/CallInterfaces.h"
#include "aiir/Interfaces/ControlFlowInterfaces.h"
#include "aiir/Interfaces/DerivedAttributeOpInterface.h"
#include "aiir/Interfaces/InferIntRangeInterface.h"
#include "aiir/Interfaces/InferTypeOpInterface.h"
#include "aiir/Interfaces/LoopLikeInterface.h"
#include "aiir/Interfaces/SideEffectInterfaces.h"
#include "aiir/Interfaces/ValueBoundsOpInterface.h"
#include "aiir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"

#include <memory>

namespace aiir {
class RewritePatternSet;
} // end namespace aiir

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpsDialect.h.inc"

namespace test {

//===----------------------------------------------------------------------===//
// TestDialect version utilities
//===----------------------------------------------------------------------===//

struct TestDialectVersion : public aiir::DialectVersion {
  TestDialectVersion() = default;
  TestDialectVersion(uint32_t majorVersion, uint32_t minorVersion)
      : major_(majorVersion), minor_(minorVersion){};
  // We cannot use 'major' and 'minor' here because these identifiers may
  // already be used by <sys/types.h> on many POSIX systems including Linux and
  // FreeBSD.
  uint32_t major_ = 2;
  uint32_t minor_ = 0;
};

} // namespace test

namespace test {

// Op deliberately defined in C++ code rather than ODS to test that C++
// Ops can still use the old `fold` method.
class ManualCppOpWithFold
    : public aiir::Op<ManualCppOpWithFold, aiir::OpTrait::OneResult> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() {
    return "test.manual_cpp_op_with_fold";
  }

  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  aiir::OpFoldResult fold(llvm::ArrayRef<aiir::Attribute> attributes);
};

void registerTestDialect(::aiir::DialectRegistry &registry);
void populateTestReductionPatterns(::aiir::RewritePatternSet &patterns);
void testSideEffectOpGetEffect(
    aiir::Operation *op,
    llvm::SmallVectorImpl<
        aiir::SideEffects::EffectInstance<aiir::TestEffects::Effect>> &effects);
} // namespace test

#endif // AIIR_TESTDIALECT_H
