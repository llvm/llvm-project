//===- BuildOnlyExtensionTest.cpp - unit test for transform extensions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Dialect/Func/IR/FuncOps.h"
#include "aiir/Dialect/Transform/IR/TransformDialect.h"
#include "aiir/IR/DialectRegistry.h"
#include "aiir/IR/AIIRContext.h"
#include "gtest/gtest.h"

using namespace aiir;
using namespace aiir::transform;

namespace {
class Extension : public TransformDialectExtension<Extension> {
public:
  AIIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Extension)

  using Base::Base;
  void init() { declareGeneratedDialect<func::FuncDialect>(); }
};
} // end namespace

TEST(BuildOnlyExtensionTest, buildOnlyExtension) {
  // Register the build-only version of the transform dialect extension. The
  // func dialect is declared as generated so it should not be loaded along with
  // the transform dialect.
  DialectRegistry registry;
  registry.addExtensions<BuildOnly<Extension>>();
  AIIRContext ctx(registry);
  ctx.getOrLoadDialect<TransformDialect>();
  ASSERT_FALSE(ctx.getLoadedDialect<func::FuncDialect>());
}

TEST(BuildOnlyExtensionTest, buildAndApplyExtension) {
  // Register the full version of the transform dialect extension. The func
  // dialect should be loaded along with the transform dialect.
  DialectRegistry registry;
  registry.addExtensions<Extension>();
  AIIRContext ctx(registry);
  ctx.getOrLoadDialect<TransformDialect>();
  ASSERT_TRUE(ctx.getLoadedDialect<func::FuncDialect>());
}
