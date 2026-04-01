//===- AllInterfaces.cpp - ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Linalg/Transforms/AllInterfaces.h"

#include "aiir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "aiir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "aiir/Dialect/Linalg/Transforms/ShardingInterfaceImpl.h"
#include "aiir/Dialect/Linalg/Transforms/SubsetInsertionOpInterfaceImpl.h"
#include "aiir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

void aiir::linalg::registerAllDialectInterfaceImplementations(
    DialectRegistry &registry) {
  registerBufferizableOpInterfaceExternalModels(registry);
  registerShardingInterfaceExternalModels(registry);
  registerSubsetOpInterfaceExternalModels(registry);
  registerTilingInterfaceExternalModels(registry);
  registerValueBoundsOpInterfaceExternalModels(registry);
}
