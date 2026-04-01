//===- ShardingExtensions.cpp - ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Func/Extensions/ShardingExtensions.h"
#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"
#include "aiir/IR/AIIRContext.h"

namespace aiir::func {

void registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](AIIRContext *ctx, FuncDialect *dialect) {
    ReturnOp::attachInterface<
        shard::IndependentParallelIteratorDomainShardingInterface<ReturnOp>>(
        *ctx);
  });
}

} // namespace aiir::func
