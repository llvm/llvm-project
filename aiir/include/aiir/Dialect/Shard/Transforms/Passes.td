//===-- Passes.td - Shard transformation definition file ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef AIIR_DIALECT_SHARD_TRANSFORMS_PASSES_TD
#define AIIR_DIALECT_SHARD_TRANSFORMS_PASSES_TD

include "aiir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShardingPropagation
//===----------------------------------------------------------------------===//

def ShardingPropagation : InterfacePass<"sharding-propagation", "aiir::FunctionOpInterface"> {
  let summary = "sharding propagation";
  let description = [{
    Propagates sharding information throughout the graph. After this pass, each
    of the operations' operands and results is annotated with a `shard.shard`
    operation, and the operations themselves are added with sharding option
    attributes.
  }];
  let options = [
    Option<"traversal", "traversal",
           "aiir::shard::TraversalOrder", /*default=*/"aiir::shard::TraversalOrder::BackwardForward",
           "Traversal order to use for sharding propagation:",
            [{::llvm::cl::values(
              clEnumValN(aiir::shard::TraversalOrder::Forward, "forward",
              "Forward only traversal."),
              clEnumValN(aiir::shard::TraversalOrder::Backward, "backward",
              "backward only traversal."),
              clEnumValN(aiir::shard::TraversalOrder::ForwardBackward, "forward-backward",
              "forward-backward traversal."),
              clEnumValN(aiir::shard::TraversalOrder::BackwardForward, "backward-forward",
              "backward-forward traversal.")
            )}]>,
  ];
  let dependentDialects = [
    "shard::ShardDialect"
  ];
}

def ShardSimplify : Pass<"shard-simplify"> {
  let summary = "Shard simplify patterns.";
  let description = [{
    Applies simplification patterns on the Shard dialect operations.
    This includes:
    - All-reduce endomorphism simplification, e.g. transforming
      `all_reduce_sum(x) + all_reduce_sum(y)` into `all_reduce_sum(x + y)`.
    - Folding `AllSliceOp(AllReduceOp)` into `ReduceScatterOp` when both ops
      share the same grid and grid_axes.
    - Folding static grid shapes into constants.
  }];
  let dependentDialects = [
    "arith::ArithDialect",
    "shard::ShardDialect"
  ];
}

def Partition : InterfacePass<"shard-partition", "aiir::FunctionOpInterface"> {
  let summary = "Partition a function into SPMD form.";
  let description = [{
    This pass fits in right after a pass that annotates the function with
    shardings like the `ShardingPropagation` pass.
    It operates on a fully annotated IR.

    A fully annotated IR required that all ranked tensor operands, results and
    block arguments are annotated with the `shard.shard` operation.
  
    All direct descendant operations in the function must implement the
    `ShardingInterface` interface or all their ranked tensor operands and
    results must have full replication sharding.

    The input IR must have sharding annotations such that each operation
    that implements `ShardingInterface` can handle during partition with
    its `partition` method.
    This can be achieved with the `ShardingPropagation` pass.

    If the function has multiple terminating blocks,
    it is the responsibility of the the one who annotates the function with
    shardings to make sure that all returns would be consisted that is,
    have the same sharding.

    Example:
    ```aiir
    shard.grid @grid_1d(shape = 2)

    func.func @f(
      %arg0: tensor<2xi8>
    ) -> tensor<2xi8> {
      %0 = shard.shard %arg0 to <@grid_1d, [[0]]> : tensor<2xi8>
      %1 = shard.shard %0 to <@grid_1d, [[0]]> annotate_for_users: tensor<2xi8>
      %2 = tosa.abs %1 : (tensor<2xi8>) -> tensor<2xi8>
      %3 = shard.shard %2 to <@grid_1d, [[0]]> : tensor<2xi8>
      %4 = shard.shard %3 to <@grid_1d, [[]]> annotate_for_users: tensor<2xi8>
      return %4 : tensor<2xi8>
    }
    ```
    Partitioning the above would result in 
    * Performing the element-wise `abs` operation on each device.
    * Resharding to full replication with an all-gather.

    ```aiir
    shard.grid @grid_1d(shape = 2)
  
    func.func @f(%arg0: tensor<1xi8>) -> tensor<2xi8> {
      %0 = tosa.abs %arg0 : (tensor<1xi8>) -> tensor<1xi8>
      %1 = shard.all_gather %0 on @grid_1d grid_axes = [0] gather_axis = 0 : tensor<1xi8> -> tensor<2xi8>
      return %1 : tensor<2xi8>
    }
    ```
  }];
  let dependentDialects = [
    "shard::ShardDialect"
  ];
}

#endif // AIIR_DIALECT_SHARD_TRANSFORMS_PASSES_TD
