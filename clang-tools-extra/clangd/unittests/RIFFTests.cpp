//===-- RIFFTests.cpp - Binary container unit tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RIFF.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(RIFFTest, File) {
  riff::File File{riff::fourCC("test"),
                  {
                      {riff::fourCC("even"), "abcd"},
                      {riff::fourCC("oddd"), "abcde"},
                  }};
  llvm::StringRef Serialized = llvm::StringRef("RIFF\x1e\0\0\0test"
                                               "even\x04\0\0\0abcd"
                                               "oddd\x05\0\0\0abcde\0",
                                               38);

  EXPECT_EQ(llvm::to_string(File), Serialized);
  auto Parsed = riff::readFile(Serialized);
  ASSERT_TRUE(bool(Parsed)) << Parsed.takeError();
  EXPECT_EQ(*Parsed, File);
}

} // namespace
} // namespace clangd
} // namespace clang
