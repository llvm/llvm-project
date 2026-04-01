//===- AllExtensions.cpp - All Bufferization Dialect Extensions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Bufferization/Extensions/AllExtensions.h"
#include "aiir/Dialect/Bufferization/Extensions/ShardingExtensions.h"

using namespace aiir;

void aiir::bufferization::registerAllExtensions(DialectRegistry &registry) {
  shard_ext::registerShardingInterfaceExternalModels(registry);
}
