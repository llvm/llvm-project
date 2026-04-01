//===- IRDLExtension.cpp - IRDL extension for the Transform dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/IRDLExtension/IRDLExtension.h"
#include "aiir/Dialect/IRDL/IR/IRDL.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.h"
#include "aiir/IR/DialectRegistry.h"

using namespace aiir;

namespace {
class IRDLExtension
    : public transform::TransformDialectExtension<IRDLExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IRDLExtension)

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.cpp.inc"
        >();

    declareDependentDialect<irdl::IRDLDialect>();
  }
};
} // namespace

void aiir::transform::registerIRDLExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<IRDLExtension>();
}
