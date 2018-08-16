//===-- FileDistanceTests.cpp  ------------------------*- C++ -*-----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileDistance.h"
#include "TestFS.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(FileDistanceTests, Distance) {
  FileDistanceOptions Opts;
  Opts.UpCost = 5;
  Opts.DownCost = 3;
  SourceParams CostTwo;
  CostTwo.Cost = 2;
  FileDistance D(
      {{"tools/clang/lib/Format/FormatToken.cpp", SourceParams()},
       {"tools/clang/include/clang/Format/FormatToken.h", SourceParams()},
       {"include/llvm/ADT/StringRef.h", CostTwo}},
      Opts);

  // Source
  EXPECT_EQ(D.distance("tools/clang/lib/Format/FormatToken.cpp"), 0u);
  EXPECT_EQ(D.distance("include/llvm/ADT/StringRef.h"), 2u);
  // Parent
  EXPECT_EQ(D.distance("tools/clang/lib/Format/"), 5u);
  // Child
  EXPECT_EQ(D.distance("tools/clang/lib/Format/FormatToken.cpp/Oops"), 3u);
  // Ancestor (up+up+up+up)
  EXPECT_EQ(D.distance("/"), 22u);
  // Sibling (up+down)
  EXPECT_EQ(D.distance("tools/clang/lib/Format/AnotherFile.cpp"), 8u);
  // Cousin (up+up+down+down)
  EXPECT_EQ(D.distance("include/llvm/Support/Allocator.h"), 18u);
  // First cousin, once removed (up+up+up+down+down)
  EXPECT_EQ(D.distance("include/llvm-c/Core.h"), 23u);
}

TEST(FileDistanceTests, BadSource) {
  // We mustn't assume that paths above sources are best reached via them.
  FileDistanceOptions Opts;
  Opts.UpCost = 5;
  Opts.DownCost = 3;
  SourceParams CostLots;
  CostLots.Cost = 100;
  FileDistance D({{"a", SourceParams()}, {"b/b/b", CostLots}}, Opts);
  EXPECT_EQ(D.distance("b"), 8u);        // a+up+down, not b+up+up
  EXPECT_EQ(D.distance("b/b/b"), 14u);   // a+up+down+down+down, not b
  EXPECT_EQ(D.distance("b/b/b/c"), 17u); // a+up+down+down+down+down, not b+down
}

// Force the unittest URI scheme to be linked,
static int LLVM_ATTRIBUTE_UNUSED UseUnittestScheme = UnittestSchemeAnchorSource;

TEST(FileDistanceTests, URI) {
  FileDistanceOptions Opts;
  Opts.UpCost = 5;
  Opts.DownCost = 3;
  SourceParams CostLots;
  CostLots.Cost = 1000;

  URIDistance D({{testPath("foo"), CostLots},
                 {"/not/a/testpath", SourceParams()},
                 {"C:\\not\\a\\testpath", SourceParams()}},
                Opts);
#ifdef _WIN32
  EXPECT_EQ(D.distance("file:///C%3a/not/a/testpath/either"), 3u);
#else
  EXPECT_EQ(D.distance("file:///not/a/testpath/either"), 3u);
#endif
  EXPECT_EQ(D.distance("unittest:///foo"), 1000u);
  EXPECT_EQ(D.distance("unittest:///bar"), 1008u);
}

TEST(FileDistance, LimitUpTraversals) {
  FileDistanceOptions Opts;
  Opts.UpCost = Opts.DownCost = 1;
  SourceParams CheapButLimited, CostLots;
  CheapButLimited.MaxUpTraversals = 1;
  CostLots.Cost = 100;

  FileDistance D({{"/", CostLots}, {"/a/b/c", CheapButLimited}}, Opts);
  EXPECT_EQ(D.distance("/a"), 101u);
  EXPECT_EQ(D.distance("/a/z"), 102u);
  EXPECT_EQ(D.distance("/a/b"), 1u);
  EXPECT_EQ(D.distance("/a/b/z"), 2u);
}

} // namespace
} // namespace clangd
} // namespace clang
