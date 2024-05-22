//===-- RecordTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Types.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

// Matches an Include* on the specified line;
MATCHER_P(line, N, "") { return arg->Line == (unsigned)N; }

TEST(RecordedIncludesTest, Match) {
  // We're using synthetic data, but need a FileManager to obtain FileEntry*s.
  // Ensure it doesn't do any actual IO.
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager FM(FileSystemOptions{});
  FileEntryRef A = FM.getVirtualFileRef("/path/a", /*Size=*/0, time_t{});
  FileEntryRef B = FM.getVirtualFileRef("/path/b", /*Size=*/0, time_t{});

  Includes Inc;
  Inc.add(Include{"a", A, SourceLocation(), 1});
  Inc.add(Include{"a2", A, SourceLocation(), 2});
  Inc.add(Include{"b", B, SourceLocation(), 3});
  Inc.add(Include{"vector", B, SourceLocation(), 4});
  Inc.add(Include{"vector", B, SourceLocation(), 5});
  Inc.add(Include{"missing", std::nullopt, SourceLocation(), 6});

  EXPECT_THAT(Inc.match(A), ElementsAre(line(1), line(2)));
  EXPECT_THAT(Inc.match(B), ElementsAre(line(3), line(4), line(5)));
  EXPECT_THAT(Inc.match(*tooling::stdlib::Header::named("<vector>")),
              ElementsAre(line(4), line(5)));
}

TEST(RecordedIncludesTest, MatchVerbatim) {
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager FM(FileSystemOptions{});
  Includes Inc;

  // By default, a verbatim header only matches includes with the same spelling.
  auto Foo =
      FM.getVirtualFileRef("repo/lib/include/rel/foo.h", /*Size=*/0, time_t{});
  Inc.add(Include{"lib/include/rel/foo.h", Foo, SourceLocation(), 1});
  Inc.add(Include{"rel/foo.h", Foo, SourceLocation(), 2});
  EXPECT_THAT(Inc.match(Header("<rel/foo.h>")), ElementsAre(line(2)));

  // A verbatim header can match another spelling if the search path
  // suggests it's equivalent.
  auto Bar =
      FM.getVirtualFileRef("repo/lib/include/rel/bar.h", /*Size=*/0, time_t{});
  Inc.addSearchDirectory("repo/");
  Inc.addSearchDirectory("repo/lib/include");
  Inc.add(Include{"lib/include/rel/bar.h", Bar, SourceLocation(), 3});
  Inc.add(Include{"rel/bar.h", Bar, SourceLocation(), 4});
  EXPECT_THAT(Inc.match(Header("<rel/bar.h>")),
              UnorderedElementsAre(line(3), line(4)));

  // We don't apply this logic to system headers, though.
  auto Vector =
      FM.getVirtualFileRef("repo/lib/include/vector", /*Size=*/0, time_t{});
  Inc.add(Include{"lib/include/vector", Vector, SourceLocation(), 5});
  EXPECT_THAT(Inc.match(Header(*tooling::stdlib::Header::named("<vector>"))),
              IsEmpty());
}

TEST(RecordedIncludesTest, MatchVerbatimMixedAbsoluteRelative) {
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FS->setCurrentWorkingDirectory("/working");
  FileManager FM(FileSystemOptions{});
  Includes Inc;

  auto Foo =
      FM.getVirtualFileRef("/working/rel1/rel2/foo.h", /*Size=*/0, time_t{});
  Inc.addSearchDirectory("rel1");
  Inc.addSearchDirectory("rel1/rel2");
  Inc.add(Include{"rel2/foo.h", Foo, SourceLocation(), 1});
  EXPECT_THAT(Inc.match(Header("<foo.h>")), IsEmpty());

  Inc = Includes{};
  auto Bar = FM.getVirtualFileRef("rel1/rel2/bar.h", /*Size=*/0, time_t{});
  Inc.addSearchDirectory("/working/rel1");
  Inc.addSearchDirectory("/working/rel1/rel2");
  Inc.add(Include{"rel2/bar.h", Bar, SourceLocation(), 1});
  EXPECT_THAT(Inc.match(Header("<bar.h>")), IsEmpty());
}

} // namespace
} // namespace clang::include_cleaner
