//===- SubsetInsertionOpInterfaceImpl.cpp - Tensor subsets ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"

#include "aiir/Dialect/Linalg/IR/Linalg.h"
#include "aiir/Interfaces/SubsetOpInterface.h"

using namespace aiir;
using namespace aiir::linalg;

namespace {
struct LinalgCopyOpSubsetOpInterface
    : public SubsetOpInterface::ExternalModel<LinalgCopyOpSubsetOpInterface,
                                              linalg::CopyOp> {
  bool operatesOnEquivalentSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // linalg.copy operates on the entire destination tensor.
    if (auto otherCopyOp = dyn_cast<linalg::CopyOp>(candidate.getOperation()))
      return equivalenceFn(cast<linalg::CopyOp>(op).getOutputs()[0],
                           otherCopyOp.getOutputs()[0]);
    // In the absence of an analysis, "false" is a conservative way to implement
    // this interface.
    return false;
  }

  bool operatesOnDisjointSubset(
      Operation *op, SubsetOpInterface candidate,
      function_ref<bool(Value, Value)> equivalenceFn) const {
    // In the absence of an analysis, "false" is a conservative way to implement
    // this interface.
    return false;
  }
};

struct LinalgCopyOpInterface
    : public SubsetInsertionOpInterface::ExternalModel<LinalgCopyOpInterface,
                                                       linalg::CopyOp> {
  OpOperand &getSourceOperand(Operation *op) const {
    auto copyOp = cast<CopyOp>(op);
    return llvm::getSingleElement(copyOp.getInputsMutable());
  }

  bool
  isEquivalentSubset(Operation *op, Value candidate,
                     function_ref<bool(Value, Value)> equivalenceFn) const {
    auto copyOp = cast<CopyOp>(op);
    return equivalenceFn(candidate,
                         llvm::getSingleElement(copyOp.getOutputs()));
  }

  Value buildSubsetExtraction(Operation *op, OpBuilder &builder,
                              Location loc) const {
    auto copyOp = cast<CopyOp>(op);
    return llvm::getSingleElement(copyOp.getOutputs());
  }

  SmallVector<Value>
  getValuesNeededToBuildSubsetExtraction(Operation *op) const {
    auto copyOp = cast<CopyOp>(op);
    return {llvm::getSingleElement(copyOp.getOutputs())};
  }
};
} // namespace

void aiir::linalg::registerSubsetOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, linalg::LinalgDialect *dialect) {
    linalg::CopyOp::attachInterface<LinalgCopyOpSubsetOpInterface>(*ctx);
    linalg::CopyOp::attachInterface<LinalgCopyOpInterface>(*ctx);
  });
}
