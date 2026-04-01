//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tosa/IR/ShardingInterfaceImpl.h"
#include "aiir/Dialect/Shard/IR/ShardOps.h"
#include "aiir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "aiir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "aiir/Dialect/Tosa/IR/TosaOps.h"
#include "aiir/IR/AffineMap.h"
#include "aiir/IR/DialectRegistry.h"

#define DEBUG_TYPE "tosa-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace aiir;
using namespace aiir::tosa;
using namespace aiir::shard;

namespace {

// loop types: [parallel, parallel, parallel, reduction_sum]
// indexing maps:
// (d0, d1, d2, d3) -> (d0, d1, d3)
// (d0, d1, d2, d3) -> (d0, d3, d2)
// (d0, d1, d2, d3) -> (d0, d1, d2)
struct MatMulOpSharding
    : public ShardingInterface::ExternalModel<MatMulOpSharding, MatMulOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType)
      return {};

    SmallVector<utils::IteratorType> types(tensorType.getRank() + 1,
                                           utils::IteratorType::parallel);
    types[tensorType.getRank()] = utils::IteratorType::reduction;
    return types;
  }

  SmallVector<ReductionKind>
  getReductionLoopIteratorKinds(Operation *op) const {
    return SmallVector<ReductionKind>(1, ReductionKind::Sum);
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!tensorType)
      return {};
    AIIRContext *ctx = op->getContext();
    SmallVector<AffineMap> maps;
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx));
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx));
    maps.push_back(AffineMap::get(0, 0, {}, ctx));
    maps.push_back(AffineMap::get(0, 0, {}, ctx));
    maps.push_back(AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx));
    return maps;
  }
};

struct NegateOpSharding
    : public ShardingInterface::ExternalModel<NegateOpSharding, NegateOp> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                           utils::IteratorType::parallel);
    return types;
  }

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    AIIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    int64_t rank = type.getRank();
    SmallVector<AffineMap> maps = {
        AffineMap::getMultiDimIdentityMap(rank, ctx),
        AffineMap::get(0, 0, {}, ctx), AffineMap::get(0, 0, {}, ctx),
        AffineMap::getMultiDimIdentityMap(rank, ctx)};
    return maps;
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitiondOperands,
                          ArrayRef<Sharding> operandShardings,
                          ArrayRef<Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTable,
                          OpBuilder &builder) const {
    partitionTriviallyShardableOperation(*op, partitiondOperands,
                                         operandShardings, resultShardings,
                                         partitionMap, symbolTable, builder);
    return success();
  }
};

template <typename OpType>
static void registerElemwiseOne(AIIRContext *ctx) {
  OpType::template attachInterface<ElementwiseShardingInterface<OpType>>(*ctx);
}

/// Variadic helper function.
template <typename... OpTypes>
static void registerElemwiseAll(AIIRContext *ctx) {
  (registerElemwiseOne<OpTypes>(ctx), ...);
}

} // namespace

void aiir::tosa::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](AIIRContext *ctx, TosaDialect *dialect) {
    registerElemwiseAll<
        ClampOp, SigmoidOp, TanhOp, AddOp, ArithmeticRightShiftOp, BitwiseAndOp,
        BitwiseOrOp, BitwiseXorOp, IntDivOp, LogicalAndOp, LogicalLeftShiftOp,
        LogicalRightShiftOp, LogicalOrOp, LogicalXorOp, MaximumOp, MinimumOp,
        MulOp, PowOp, SubOp, AbsOp, BitwiseNotOp, CeilOp, ClzOp, ExpOp, FloorOp,
        LogOp, LogicalNotOp, ReciprocalOp, RsqrtOp, SelectOp, EqualOp,
        GreaterOp, GreaterEqualOp>(ctx);

    MatMulOp::attachInterface<MatMulOpSharding>(*ctx);
    NegateOp::attachInterface<NegateOpSharding>(*ctx);
  });
}
