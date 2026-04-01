//===- ShardingExtensions.h -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_BUFFERIZATION_SHARDINGEXTENSIONS_H
#define AIIR_DIALECT_BUFFERIZATION_SHARDINGEXTENSIONS_H

namespace aiir {
class DialectRegistry;

namespace bufferization {
namespace shard_ext {
void registerShardingInterfaceExternalModels(DialectRegistry &registry);
} // namespace shard_ext
} // namespace bufferization
} // namespace aiir

#endif // AIIR_DIALECT_BUFFERIZATION_SHARDINGEXTENSIONS_H
