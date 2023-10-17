//===-- CanonicalIncludesTests.cpp - --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "index/CanonicalIncludes.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
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

TEST(CanonicalIncludesTest, SystemHeaderMap) {
  auto InMemFS = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  FileManager Files(FileSystemOptions(), InMemFS);

  CanonicalIncludes CI;
  LangOptions Language;
  Language.CPlusPlus = true;
  CI.addSystemHeadersMapping(Language);

  // We should have a path from 'bits/stl_vector.h' to '<vector>'.
  // FIXME: The Standrad Library map in CanonicalIncludes expects forward
  // slashes and Windows would use backward slashes instead, so the headers are
  // not matched appropriately.
  auto STLVectorFile = addFile(*InMemFS, Files, "bits/stl_vector.h");
  ASSERT_EQ("<vector>", CI.mapHeader(STLVectorFile.getName()));
}

} // namespace
} // namespace clangd
} // namespace clang
