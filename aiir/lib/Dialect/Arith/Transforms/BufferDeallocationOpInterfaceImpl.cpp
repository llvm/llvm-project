//===- BufferDeallocationOpInterfaceImpl.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "aiir/Dialect/Arith/IR/Arith.h"
#include "aiir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "aiir/Dialect/MemRef/IR/MemRef.h"
#include "aiir/IR/Dialect.h"
#include "aiir/IR/Operation.h"

using namespace aiir;
using namespace aiir::bufferization;

namespace {
/// Provides custom logic to materialize ownership indicator values for the
/// result value of 'arith.select'. Instead of cloning or runtime alias
/// checking, this implementation inserts another `arith.select` to choose the
/// ownership indicator of the operand in the same way the original
/// `arith.select` chooses the MemRef operand. If at least one of the operand's
/// ownerships is 'Unknown', fall back to the default implementation.
///
/// Example:
/// ```aiir
/// // let ownership(%m0) := %o0
/// // let ownership(%m1) := %o1
/// %res = arith.select %cond, %m0, %m1
/// ```
/// The default implementation would insert a clone and replace all uses of the
/// result of `arith.select` with that clone:
/// ```aiir
/// %res = arith.select %cond, %m0, %m1
/// %clone = bufferization.clone %res
/// // let ownership(%res) := 'Unknown'
/// // let ownership(%clone) := %true
/// // replace all uses of %res with %clone
/// ```
/// This implementation, on the other hand, materializes the following:
/// ```aiir
/// %res = arith.select %cond, %m0, %m1
/// %res_ownership = arith.select %cond, %o0, %o1
/// // let ownership(%res) := %res_ownership
/// ```
struct SelectOpInterface
    : public BufferDeallocationOpInterface::ExternalModel<SelectOpInterface,
                                                          arith::SelectOp> {
  FailureOr<Operation *> process(Operation *op, DeallocationState &state,
                                 const DeallocationOptions &options) const {
    return op; // nothing to do
  }

  std::pair<Value, Value>
  materializeUniqueOwnershipForMemref(Operation *op, DeallocationState &state,
                                      const DeallocationOptions &options,
                                      OpBuilder &builder, Value value) const {
    auto selectOp = cast<arith::SelectOp>(op);
    assert(value == selectOp.getResult() &&
           "Value not defined by this operation");

    Block *block = value.getParentBlock();
    if (!state.getOwnership(selectOp.getTrueValue(), block).isUnique() ||
        !state.getOwnership(selectOp.getFalseValue(), block).isUnique())
      return state.getMemrefWithUniqueOwnership(builder, value,
                                                value.getParentBlock());

    Value ownership = arith::SelectOp::create(
        builder, op->getLoc(), selectOp.getCondition(),
        state.getOwnership(selectOp.getTrueValue(), block).getIndicator(),
        state.getOwnership(selectOp.getFalseValue(), block).getIndicator());
    return {selectOp.getResult(), ownership};
  }
};

} // namespace

void aiir::arith::registerBufferDeallocationOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, ArithDialect *dialect) {
    SelectOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
