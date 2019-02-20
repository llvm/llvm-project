//===-- RIFFTests.cpp - Binary container unit tests -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RIFF.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace {
using ::testing::ElementsAre;

TEST(RIFFTest, File) {
  riff::File File{riff::fourCC("test"),
                  {
                      {riff::fourCC("even"), "abcd"},
                      {riff::fourCC("oddd"), "abcde"},
                  }};
  StringRef Serialized = StringRef("RIFF\x1e\0\0\0test"
                                   "even\x04\0\0\0abcd"
                                   "oddd\x05\0\0\0abcde\0",
                                   38);

  EXPECT_EQ(to_string(File), Serialized);
  auto Parsed = riff::readFile(Serialized);
  ASSERT_TRUE(bool(Parsed)) << Parsed.takeError();
  EXPECT_EQ(*Parsed, File);
}

} // namespace
} // namespace clangd
} // namespace clang
