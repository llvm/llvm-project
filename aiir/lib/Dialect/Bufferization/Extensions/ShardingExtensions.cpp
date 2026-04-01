//===- ShardingExtensions.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Bufferization/Extensions/ShardingExtensions.h"
#include "aiir/Dialect/Bufferization/IR/Bufferization.h"
#include "aiir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "aiir/IR/DialectRegistry.h"

using namespace aiir;
using namespace aiir::bufferization;
using namespace aiir::shard;

void aiir::bufferization::shard_ext::registerShardingInterfaceExternalModels(
    DialectRegistry &registry) {

  registry.addExtension(+[](AIIRContext *ctx, BufferizationDialect *dialect) {
    AllocTensorOp::attachInterface<
        IndependentParallelIteratorDomainShardingInterface<AllocTensorOp>>(
        *ctx);
    DeallocTensorOp::attachInterface<
        IndependentParallelIteratorDomainShardingInterface<DeallocTensorOp>>(
        *ctx);
    MaterializeInDestinationOp::attachInterface<
        ElementwiseShardingInterface<MaterializeInDestinationOp>>(*ctx);
  });
}
