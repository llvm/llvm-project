//===-- Unittests for StringStream ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/stringstream.h"
#include "test/UnitTest/Test.h"

using __llvm_libc::cpp::span;
using __llvm_libc::cpp::StringStream;

TEST(LlvmLibcStringStreamTest, Simple) {
  char buf[256];

  StringStream ss1(buf);
  ss1 << "Hello, Stream - " << int(123) << StringStream::ENDS;
  ASSERT_FALSE(ss1.overflow());
  ASSERT_STREQ(ss1.str().data(), "Hello, Stream - 123");

  StringStream ss2(buf);
  ss2 << 'a' << 'b' << 'c' << StringStream::ENDS;
  ASSERT_FALSE(ss2.overflow());
  ASSERT_STREQ(ss2.str().data(), "abc");
}

TEST(LlvmLibcStringStreamTest, Overflow) {
  constexpr size_t BUFSIZE = 8;
  char buf[BUFSIZE];

  StringStream ss1(buf);
  ss1 << "Hello, Stream - " << int(123) << StringStream::ENDS;
  ASSERT_TRUE(ss1.overflow());
  ASSERT_EQ(ss1.str().size(), BUFSIZE);

  StringStream ss2(buf);
  ss2 << "7777777";
  ASSERT_FALSE(ss2.overflow());
  ASSERT_EQ(ss2.str().size(), size_t(7));
  ss2 << "8";
  ASSERT_FALSE(ss2.overflow());
  ASSERT_EQ(ss2.str().size(), size_t(8));
  ss2 << StringStream::ENDS;
  ASSERT_TRUE(ss2.overflow());
  ASSERT_EQ(ss2.str().size(), BUFSIZE);
}
