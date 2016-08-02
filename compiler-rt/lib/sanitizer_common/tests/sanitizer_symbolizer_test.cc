//===-- sanitizer_symbolizer_test.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tests for sanitizer_symbolizer.h and sanitizer_symbolizer_internal.h
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_symbolizer_internal.h"
#include "gtest/gtest.h"

namespace __sanitizer {

TEST(Symbolizer, ExtractToken) {
  char *token;
  const char *rest;

  rest = ExtractToken("a;b;c", ";", &token);
  EXPECT_STREQ("a", token);
  EXPECT_STREQ("b;c", rest);
  InternalFree(token);

  rest = ExtractToken("aaa-bbb.ccc", ";.-*", &token);
  EXPECT_STREQ("aaa", token);
  EXPECT_STREQ("bbb.ccc", rest);
  InternalFree(token);
}

TEST(Symbolizer, ExtractInt) {
  int token;
  const char *rest = ExtractInt("123,456;789", ";,", &token);
  EXPECT_EQ(123, token);
  EXPECT_STREQ("456;789", rest);
}

TEST(Symbolizer, ExtractUptr) {
  uptr token;
  const char *rest = ExtractUptr("123,456;789", ";,", &token);
  EXPECT_EQ(123U, token);
  EXPECT_STREQ("456;789", rest);
}

TEST(Symbolizer, ExtractTokenUpToDelimiter) {
  char *token;
  const char *rest =
      ExtractTokenUpToDelimiter("aaa-+-bbb-+-ccc", "-+-", &token);
  EXPECT_STREQ("aaa", token);
  EXPECT_STREQ("bbb-+-ccc", rest);
  InternalFree(token);
}

}  // namespace __sanitizer
