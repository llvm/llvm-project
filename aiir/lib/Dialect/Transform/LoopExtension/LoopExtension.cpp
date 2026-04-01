//===- LoopExtension.cpp - Loop extension for the Transform dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/LoopExtension/LoopExtension.h"

#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "aiir/IR/DialectRegistry.h"

using namespace aiir;

namespace {
/// Loop extension of the Transform dialect. This provides "core" transform
/// operations for loop-like ops.
class LoopExtension
    : public transform::TransformDialectExtension<LoopExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopExtension)

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Transform/LoopExtension/LoopExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void aiir::transform::registerLoopExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<LoopExtension>();
}
