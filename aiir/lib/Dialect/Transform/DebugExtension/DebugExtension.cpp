//===- DebugExtension.cpp - Debug extension for the Transform dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/DebugExtension/DebugExtension.h"

#include "aiir/Dialect/Transform/DebugExtension/DebugExtensionOps.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/IR/DialectRegistry.h"

using namespace aiir;

namespace {
/// Debug extension of the Transform dialect. This provides operations for
/// debugging transform dialect scripts.
class DebugExtension
    : public transform::TransformDialectExtension<DebugExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DebugExtension)

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Transform/DebugExtension/DebugExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void aiir::transform::registerDebugExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<DebugExtension>();
}
