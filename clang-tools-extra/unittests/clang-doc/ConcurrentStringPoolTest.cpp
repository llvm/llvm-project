//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for clang::doc::ConcurrentStringPool.
///
//===----------------------------------------------------------------------===//

#include "ConcurrentStringPool.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Parallel.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace llvm;
using namespace clang::doc;

namespace {

TEST(ConcurrentStringPoolTest, SaveOverloadsAndEmpty) {
  ConcurrentStringPool Pool;

  EXPECT_TRUE(Pool.save("").empty());
  EXPECT_TRUE(Pool.save(StringRef()).empty());
  EXPECT_TRUE(Pool.save(std::string()).empty());
  EXPECT_TRUE(Pool.save(Twine()).empty());
  EXPECT_TRUE(Pool.save(Twine("")).empty());
  EXPECT_TRUE(Pool.save(Twine("") + Twine("")).empty());

  const char *CStr = "hello_world";
  std::string StdStr = "hello_world";
  StringRef RefStr = "hello_world";

  StringRef SavedCStr = Pool.save(CStr);
  StringRef SavedStd = Pool.save(StdStr);
  StringRef SavedRef = Pool.save(RefStr);
  StringRef SavedSingleTwine = Pool.save(Twine(RefStr));
  StringRef SavedConcatTwine = Pool.save(Twine("hello_") + "world");

  EXPECT_EQ(SavedCStr, "hello_world");
  EXPECT_EQ(SavedStd.data(), SavedCStr.data());
  EXPECT_EQ(SavedRef.data(), SavedCStr.data());
  EXPECT_EQ(SavedSingleTwine.data(), SavedCStr.data());
  EXPECT_EQ(SavedConcatTwine.data(), SavedCStr.data());
}

TEST(ConcurrentStringPoolTest, AddStringEntries) {
  ConcurrentStringPool Pool;

  StringRef Res1 = Pool.save("1");
  EXPECT_EQ(Res1, "1");

  StringRef Res2 = Pool.save("2");
  EXPECT_EQ(Res1, "1");
  EXPECT_EQ(Res2, "2");
  EXPECT_NE(Res1.data(), Res2.data());

  StringRef Res3 = Pool.save("3");
  EXPECT_EQ(Res3, "3");
  EXPECT_NE(Res1.data(), Res3.data());
  EXPECT_NE(Res2.data(), Res3.data());

  StringRef Res4 = Pool.save("1");
  EXPECT_EQ(Res4, "1");
  EXPECT_EQ(Res1.data(), Res4.data());
}

TEST(ConcurrentStringPoolTest, EmbeddedNulls) {
  ConcurrentStringPool Pool;

  StringRef WithNull1("foo\0bar", 7);
  StringRef WithNull2("foo\0baz", 7);
  StringRef Prefix("foo", 3);

  StringRef Saved1 = Pool.save(WithNull1);
  StringRef Saved2 = Pool.save(WithNull2);
  StringRef SavedPrefix = Pool.save(Prefix);

  EXPECT_EQ(Saved1.size(), 7u);
  EXPECT_EQ(Saved1, WithNull1);
  EXPECT_EQ(Saved2, WithNull2);
  EXPECT_EQ(SavedPrefix, Prefix);
  EXPECT_NE(Saved1.data(), Saved2.data());
  EXPECT_NE(Saved1.data(), SavedPrefix.data());
  EXPECT_EQ(Pool.save(WithNull1).data(), Saved1.data());
  EXPECT_EQ(Pool.save(WithNull2).data(), Saved2.data());
}

TEST(ConcurrentStringPoolTest, AddStringMultipleEntriesWithSlabGrowth) {
  ConcurrentStringPoolImpl<1, 8, 2> Pool;
  constexpr size_t NumElements = 500;
  std::vector<StringRef> Inserted(NumElements);

  for (size_t I = 0; I < NumElements; ++I) {
    std::string S = formatv("{0} {1}", I, I + 100);
    Inserted[I] = Pool.save(S);
    EXPECT_EQ(Inserted[I], S);
  }

  for (size_t I = 0; I < NumElements; ++I) {
    std::string S = formatv("{0} {1}", I, I + 100);
    StringRef Dup = Pool.save(S);
    EXPECT_EQ(Dup, S);
    EXPECT_EQ(Dup.data(), Inserted[I].data());
  }
}

TEST(ConcurrentStringPoolTest, AddStringEntriesParallel) {
  ConcurrentStringPool Pool;
  constexpr size_t NumElements = 10000;
  std::vector<StringRef> Inserted(NumElements);

  parallelFor(0, NumElements, [&](size_t I) {
    std::string S = formatv("{0}", I);
    StringRef Entry = Pool.save(S);
    EXPECT_EQ(Entry, S);
    Inserted[I] = Entry;
  });

  parallelFor(0, NumElements, [&](size_t I) {
    std::string S = formatv("{0}", I);
    StringRef Entry = Pool.save(S);
    EXPECT_EQ(Entry, S);
    EXPECT_EQ(Entry.data(), Inserted[I].data());
  });
}

TEST(ConcurrentStringPoolTest, AddStringEntriesParallelWithSlabGrowth) {
  ConcurrentStringPoolImpl<6, 8, 2> Pool;
  constexpr size_t NumElements = 4000;
  std::vector<StringRef> Inserted(NumElements);

  parallelFor(0, NumElements, [&](size_t I) {
    std::string S = formatv("{0}", I);
    StringRef Entry = Pool.save(S);
    EXPECT_EQ(Entry, S);
    Inserted[I] = Entry;
  });

  parallelFor(0, NumElements, [&](size_t I) {
    std::string S = formatv("{0}", I);
    StringRef Entry = Pool.save(S);
    EXPECT_EQ(Entry, S);
    EXPECT_EQ(Entry.data(), Inserted[I].data());
  });
}

} // namespace
