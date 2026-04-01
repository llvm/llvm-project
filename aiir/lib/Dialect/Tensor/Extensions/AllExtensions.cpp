//===- AllExtensions.cpp - All Tensor Dialect Extensions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Tensor/Extensions/AllExtensions.h"
#include "aiir/Dialect/Tensor/Extensions/ShardingExtensions.h"

using namespace aiir;

void aiir::tensor::registerAllExtensions(DialectRegistry &registry) {
  registerShardingInterfaceExternalModels(registry);
}
