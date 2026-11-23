//===- unittests/Serialization/ResolveImportedPathTest.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTReader.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

std::string resolve(StringRef Path, StringRef Prefix) {
  SmallString<0> Buf;
  Buf.reserve(64);
  return ASTReader::ResolveImportedPathAndAllocate(Buf, Path, Prefix);
}

TEST(ResolveImportedPathTest, RelativeToBase) {
  EXPECT_EQ(llvm::sys::path::convert_to_slash(resolve("dir/file.h", "/base")),
            "/base/dir/file.h");
}

TEST(ResolveImportedPathTest, BaseItself) {
  EXPECT_EQ(resolve(".", "/base"), "/base");
}

} // namespace
