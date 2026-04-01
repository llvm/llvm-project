//===- ShardingExtensions.h - -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_
#define AIIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_

namespace aiir {

class DialectRegistry;

namespace func {

void registerShardingInterfaceExternalModels(DialectRegistry &registry);

} // namespace func
} // namespace aiir

#endif // AIIR_DIALECT_FUNC_IR_SHARDINGINTERFACEIMPL_H_
