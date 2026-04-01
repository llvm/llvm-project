//===- BufferDeallocationOpInterfaceImpl.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/GPU/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "aiir/Dialect/GPU/IR/GPUDialect.h"

using namespace aiir;
using namespace aiir::bufferization;

namespace {
///
struct GPUTerminatorOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<
          GPUTerminatorOpInterface, gpu::TerminatorOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    SmallVector<Value> updatedOperandOwnerships;
    return deallocation_impl::insertDeallocOpForReturnLike(
        state, op, {}, updatedOperandOwnerships);
  }
};

} // namespace

void aiir::gpu::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, GPUDialect *dialect) {
    gpu::TerminatorOp::attachInterface<GPUTerminatorOpInterface>(*ctx);
  });
}
