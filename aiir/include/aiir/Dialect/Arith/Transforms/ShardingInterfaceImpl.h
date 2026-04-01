//===- ShardingInterfaceImpl.h - ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_DIALECT_ARITH_TRANSFORMS_SHARDINGINTERFACEIMPL_H_
#define AIIR_DIALECT_ARITH_TRANSFORMS_SHARDINGINTERFACEIMPL_H_

namespace aiir {

class DialectRegistry;

namespace arith {

void registerShardingInterfaceExternalModels(DialectRegistry &registry);

} // namespace arith
} // namespace aiir

#endif // AIIR_DIALECT_ARITH_TRANSFORMS_SHARDINGINTERFACEIMPL_H_
