//===-- ProjectAwareIndexTests.cpp  -------------------*- C++ -*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "TestIndex.h"
#include "index/Index.h"
#include "index/MemIndex.h"
#include "index/ProjectAware.h"
#include "index/Ref.h"
#include "index/Relation.h"
#include "support/Context.h"
#include "support/Threading.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <utility>

namespace clang {
namespace clangd {
using testing::ElementsAre;
using testing::IsEmpty;

std::unique_ptr<SymbolIndex> createIndex() {
  SymbolSlab::Builder Builder;
  Builder.insert(symbol("1"));
  return MemIndex::build(std::move(Builder).build(), RefSlab(), RelationSlab());
}

TEST(ProjectAware, Test) {
  IndexFactory Gen = [](const Config::ExternalIndexSpec &, AsyncTaskRunner *) {
    return createIndex();
  };

  auto Idx = createProjectAwareIndex(std::move(Gen), true);
  FuzzyFindRequest Req;
  Req.Query = "1";
  Req.AnyScope = true;

  EXPECT_THAT(match(*Idx, Req), IsEmpty());

  Config C;
  C.Index.External.Kind = Config::ExternalIndexSpec::File;
  C.Index.External.Location = "test";
  WithContextValue With(Config::Key, std::move(C));
  EXPECT_THAT(match(*Idx, Req), ElementsAre("1"));
}

TEST(ProjectAware, CreatedOnce) {
  unsigned InvocationCount = 0;
  IndexFactory Gen = [&](const Config::ExternalIndexSpec &, AsyncTaskRunner *) {
    ++InvocationCount;
    return createIndex();
  };

  auto Idx = createProjectAwareIndex(std::move(Gen), true);
  // No invocation at start.
  EXPECT_EQ(InvocationCount, 0U);
  FuzzyFindRequest Req;
  Req.Query = "1";
  Req.AnyScope = true;

  // Cannot invoke without proper config.
  match(*Idx, Req);
  EXPECT_EQ(InvocationCount, 0U);

  Config C;
  C.Index.External.Kind = Config::ExternalIndexSpec::File;
  C.Index.External.Location = "test";
  WithContextValue With(Config::Key, std::move(C));
  match(*Idx, Req);
  // Now it should be created.
  EXPECT_EQ(InvocationCount, 1U);
  match(*Idx, Req);
  // It is cached afterwards.
  EXPECT_EQ(InvocationCount, 1U);
}
} // namespace clangd
} // namespace clang
