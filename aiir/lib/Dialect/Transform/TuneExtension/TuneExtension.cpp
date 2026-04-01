//===- TuneExtension.cpp - Tune extension for the Transform dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Transform/TuneExtension/TuneExtension.h"

#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/Dialect/Transform/TuneExtension/TuneExtensionOps.h"
#include "aiir/IR/DialectRegistry.h"

using namespace aiir;

class TuneExtension
    : public transform::TransformDialectExtension<TuneExtension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TuneExtension)

  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "aiir/Dialect/Transform/TuneExtension/TuneExtensionOps.cpp.inc"
        >();
  }
};

void aiir::transform::registerTuneExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<TuneExtension>();
}
