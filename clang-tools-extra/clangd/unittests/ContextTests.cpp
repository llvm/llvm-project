//===-- ContextTests.cpp - Context tests ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Context.h"

#include "gtest/gtest.h"

namespace clang {
namespace clangd {

TEST(ContextTests, Simple) {
  Key<int> IntParam;
  Key<int> ExtraIntParam;

  Context Ctx = Context::empty().derive(IntParam, 10).derive(ExtraIntParam, 20);

  EXPECT_EQ(*Ctx.get(IntParam), 10);
  EXPECT_EQ(*Ctx.get(ExtraIntParam), 20);
}

TEST(ContextTests, MoveOps) {
  Key<std::unique_ptr<int>> Param;

  Context Ctx = Context::empty().derive(Param, llvm::make_unique<int>(10));
  EXPECT_EQ(**Ctx.get(Param), 10);

  Context NewCtx = std::move(Ctx);
  EXPECT_EQ(**NewCtx.get(Param), 10);
}

TEST(ContextTests, Builders) {
  Key<int> ParentParam;
  Key<int> ParentAndChildParam;
  Key<int> ChildParam;

  Context ParentCtx =
      Context::empty().derive(ParentParam, 10).derive(ParentAndChildParam, 20);
  Context ChildCtx =
      ParentCtx.derive(ParentAndChildParam, 30).derive(ChildParam, 40);

  EXPECT_EQ(*ParentCtx.get(ParentParam), 10);
  EXPECT_EQ(*ParentCtx.get(ParentAndChildParam), 20);
  EXPECT_EQ(ParentCtx.get(ChildParam), nullptr);

  EXPECT_EQ(*ChildCtx.get(ParentParam), 10);
  EXPECT_EQ(*ChildCtx.get(ParentAndChildParam), 30);
  EXPECT_EQ(*ChildCtx.get(ChildParam), 40);
}

} // namespace clangd
} // namespace clang
