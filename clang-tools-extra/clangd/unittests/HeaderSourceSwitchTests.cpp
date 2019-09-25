//===--- HeaderSourceSwitchTests.cpp - ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HeaderSourceSwitch.h"

#include "TestFS.h"
#include "TestTU.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(HeaderSourceSwitchTest, FileHeuristic) {
  MockFSProvider FS;
  auto FooCpp = testPath("foo.cpp");
  auto FooH = testPath("foo.h");
  auto Invalid = testPath("main.cpp");

  FS.Files[FooCpp];
  FS.Files[FooH];
  FS.Files[Invalid];
  Optional<Path> PathResult =
      getCorrespondingHeaderOrSource(FooCpp, FS.getFileSystem());
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooH);

  PathResult = getCorrespondingHeaderOrSource(FooH, FS.getFileSystem());
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooCpp);

  // Test with header file in capital letters and different extension, source
  // file with different extension
  auto FooC = testPath("bar.c");
  auto FooHH = testPath("bar.HH");

  FS.Files[FooC];
  FS.Files[FooHH];
  PathResult = getCorrespondingHeaderOrSource(FooC, FS.getFileSystem());
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), FooHH);

  // Test with both capital letters
  auto Foo2C = testPath("foo2.C");
  auto Foo2HH = testPath("foo2.HH");
  FS.Files[Foo2C];
  FS.Files[Foo2HH];
  PathResult = getCorrespondingHeaderOrSource(Foo2C, FS.getFileSystem());
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo2HH);

  // Test with source file as capital letter and .hxx header file
  auto Foo3C = testPath("foo3.C");
  auto Foo3HXX = testPath("foo3.hxx");

  FS.Files[Foo3C];
  FS.Files[Foo3HXX];
  PathResult = getCorrespondingHeaderOrSource(Foo3C, FS.getFileSystem());
  EXPECT_TRUE(PathResult.hasValue());
  ASSERT_EQ(PathResult.getValue(), Foo3HXX);

  // Test if asking for a corresponding file that doesn't exist returns an empty
  // string.
  PathResult = getCorrespondingHeaderOrSource(Invalid, FS.getFileSystem());
  EXPECT_FALSE(PathResult.hasValue());
}

} // namespace
} // namespace clangd
} // namespace clang
