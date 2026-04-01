//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"

#include "aiir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "aiir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "aiir/IR/Operation.h"

using namespace aiir;
using namespace aiir::bufferization;

namespace aiir {
namespace cf {
namespace {

template <typename ConcreteModel, typename ConcreteOp>
struct BranchLikeOpInterface
    : public BranchOpBufferizableOpInterfaceExternalModel<ConcreteModel,
                                                          ConcreteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  LogicalResult verifyAnalysis(Operation *op,
                               const AnalysisState &state) const {
    return success();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    // The operands of this op are bufferized together with the block signature.
    return success();
  }
};

/// Bufferization of cf.br.
struct BranchOpInterface
    : public BranchLikeOpInterface<BranchOpInterface, cf::BranchOp> {};

/// Bufferization of cf.cond_br.
struct CondBranchOpInterface
    : public BranchLikeOpInterface<CondBranchOpInterface, cf::CondBranchOp> {};

} // namespace
} // namespace cf
} // namespace aiir

void aiir::cf::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, cf::ControlFlowDialect *dialect) {
    cf::BranchOp::attachInterface<BranchOpInterface>(*ctx);
    cf::CondBranchOp::attachInterface<CondBranchOpInterface>(*ctx);
  });
}
