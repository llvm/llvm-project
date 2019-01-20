//===- unittests/Rewrite/RewriteBufferTest.cpp - RewriteBuffer tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Rewrite/Core/RewriteBuffer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

static void tagRange(unsigned Offset, unsigned Len, StringRef tagName,
                     RewriteBuffer &Buf) {
  std::string BeginTag;
  raw_string_ostream(BeginTag) << '<' << tagName << '>';
  std::string EndTag;
  raw_string_ostream(EndTag) << "</" << tagName << '>';

  Buf.InsertTextAfter(Offset, BeginTag);
  Buf.InsertTextBefore(Offset+Len, EndTag);
}

TEST(RewriteBuffer, TagRanges) {
  StringRef Input = "hello world";
  const char *Output = "<outer><inner>hello</inner></outer> ";

  RewriteBuffer Buf;
  Buf.Initialize(Input);
  StringRef RemoveStr = "world";
  size_t Pos = Input.find(RemoveStr);
  Buf.RemoveText(Pos, RemoveStr.size());

  StringRef TagStr = "hello";
  Pos = Input.find(TagStr);
  tagRange(Pos, TagStr.size(), "outer", Buf);
  tagRange(Pos, TagStr.size(), "inner", Buf);

  std::string Result;
  raw_string_ostream OS(Result);
  Buf.write(OS);
  OS.flush();
  EXPECT_EQ(Output, Result);
}

} // anonymous namespace
