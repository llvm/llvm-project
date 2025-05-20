//===-- LLVMInsertChainFolder.h --  insertvalue chain folder ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper to fold LLVM dialect llvm.insertvalue chain representing constants
// into an Attribute representation.
// This sits in Flang because it is incomplete and tailored for flang needs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LogicalResult.h"

namespace mlir {
class Attribute;
class OpBuilder;
class Value;
} // namespace mlir

namespace fir {

/// Attempt to fold an llvm.insertvalue chain into an attribute representation
/// suitable as llvm.constant operand. The returned value will be llvm::Failure
/// if this is not an llvm.insertvalue result or if the chain is not a constant,
/// or cannot be represented as an Attribute. The operations are not deleted,
/// but some llvm.insertvalue value operands may be folded with the builder on
/// the way.
llvm::FailureOr<mlir::Attribute>
tryFoldingLLVMInsertChain(mlir::Value insertChainResult,
                          mlir::OpBuilder &builder);
} // namespace fir
