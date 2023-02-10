//===-- CanonicalIncludesTests.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFS.h"
#include "index/CanonicalIncludes.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

FileEntryRef addFile(llvm::vfs::InMemoryFileSystem &FS, FileManager &FM,
                     llvm::StringRef Filename) {
  FS.addFile(Filename, 0, llvm::MemoryBuffer::getMemBuffer(""));
  auto File = FM.getFileRef(Filename);
  EXPECT_THAT_EXPECTED(File, llvm::Succeeded());
  return *File;
}

TEST(CanonicalIncludesTest, CStandardLibrary) {
  CanonicalIncludes CI;
  auto Language = LangOptions();
  Language.C11 = true;
  CI.addSystemHeadersMapping(Language);
  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<stdio.h>", CI.mapSymbol("", "printf", Language));
  EXPECT_EQ("", CI.mapSymbol("", "unknown_symbol", Language));
}

TEST(CanonicalIncludesTest, CXXStandardLibrary) {
  CanonicalIncludes CI;
  auto Language = LangOptions();
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // Usual standard library symbols are mapped correctly.
  EXPECT_EQ("<vector>", CI.mapSymbol("std::", "vector", Language));
  EXPECT_EQ("<cstdio>", CI.mapSymbol("std::", "printf", Language));
  // std::move is ambiguous, currently always mapped to <utility>
  EXPECT_EQ("<utility>", CI.mapSymbol("std::", "move", Language));
  EXPECT_EQ("<cstddef>", CI.mapSymbol("std::", "size_t", Language));
  // Unknown std symbols aren't mapped.
  EXPECT_EQ("", CI.mapSymbol("std::", "notathing", Language));
  // iosfwd declares some symbols it doesn't own.
  EXPECT_EQ("<ostream>", CI.mapSymbol("std::", "ostream", Language));
  // And (for now) we assume it owns the others.
  auto InMemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager Files(FileSystemOptions(), InMemFS);
  auto File = addFile(*InMemFS, Files, testPath("iosfwd"));
  EXPECT_EQ("<iosfwd>", CI.mapHeader(File));
}

TEST(CanonicalIncludesTest, PathMapping) {
  auto InMemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager Files(FileSystemOptions(), InMemFS);
  std::string BarPath = testPath("foo/bar");
  auto Bar = addFile(*InMemFS, Files, BarPath);
  auto Other = addFile(*InMemFS, Files, testPath("foo/baz"));
  // As used for IWYU pragmas.
  CanonicalIncludes CI;
  CI.addMapping(Bar, "<baz>");

  // We added a mapping for baz.
  EXPECT_EQ("<baz>", CI.mapHeader(Bar));
  // Other file doesn't have a mapping.
  EXPECT_EQ("", CI.mapHeader(Other));

  // Add hard link to "foo/bar" and check that it is also mapped to <baz>, hence
  // does not depend on the header name.
  std::string HardLinkPath = testPath("hard/link");
  InMemFS->addHardLink(HardLinkPath, BarPath);
  auto HardLinkFile = Files.getFileRef(HardLinkPath);
  ASSERT_THAT_EXPECTED(HardLinkFile, llvm::Succeeded());
  EXPECT_EQ("<baz>", CI.mapHeader(*HardLinkFile));
}

TEST(CanonicalIncludesTest, Precedence) {
  auto InMemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager Files(FileSystemOptions(), InMemFS);
  auto File = addFile(*InMemFS, Files, testPath("some/path"));

  CanonicalIncludes CI;
  CI.addMapping(File, "<path>");
  LangOptions Language;
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // We added a mapping from some/path to <path>.
  ASSERT_EQ("<path>", CI.mapHeader(File));
  // We should have a path from 'bits/stl_vector.h' to '<vector>'.
  // FIXME: The Standrad Library map in CanonicalIncludes expects forward
  // slashes and Windows would use backward slashes instead, so the headers are
  // not matched appropriately.
  auto STLVectorFile = addFile(*InMemFS, Files, "bits/stl_vector.h");
  ASSERT_EQ("<vector>", CI.mapHeader(STLVectorFile));
}

} // namespace
} // namespace clangd
} // namespace clang
