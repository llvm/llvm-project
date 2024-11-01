//===-- RecordTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Types.h"
#include "clang/Basic/FileManager.h"
#include "clang/Tooling/Inclusions/StandardLibrary.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang::include_cleaner {
namespace {
using testing::ElementsAre;

// Matches an Include* on the specified line;
MATCHER_P(line, N, "") { return arg->Line == (unsigned)N; }

TEST(RecordedIncludesTest, Match) {
  // We're using synthetic data, but need a FileManager to obtain FileEntry*s.
  // Ensure it doesn't do any actual IO.
  auto FS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager FM(FileSystemOptions{});
  const FileEntry *A = FM.getVirtualFile("/path/a", /*Size=*/0, time_t{});
  const FileEntry *B = FM.getVirtualFile("/path/b", /*Size=*/0, time_t{});

  Includes Inc;
  Inc.add(Include{"a", A, SourceLocation(), 1});
  Inc.add(Include{"a2", A, SourceLocation(), 2});
  Inc.add(Include{"b", B, SourceLocation(), 3});
  Inc.add(Include{"vector", B, SourceLocation(), 4});
  Inc.add(Include{"vector", B, SourceLocation(), 5});
  Inc.add(Include{"missing", nullptr, SourceLocation(), 6});

  EXPECT_THAT(Inc.match(A), ElementsAre(line(1), line(2)));
  EXPECT_THAT(Inc.match(B), ElementsAre(line(3), line(4), line(5)));
  EXPECT_THAT(Inc.match(*tooling::stdlib::Header::named("<vector>")),
              ElementsAre(line(4), line(5)));
}

} // namespace
} // namespace clang::include_cleaner
