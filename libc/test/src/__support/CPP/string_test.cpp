//===-- Unittests for string ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/func/free.h"
#include "src/__support/CPP/string.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string;
using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::cpp::to_string;

TEST(LlvmLibcStringTest, InitializeEmpty) {
  const string s;
  ASSERT_EQ(s.size(), size_t(0));
  ASSERT_TRUE(s.empty());
  ASSERT_STREQ(s.data(), "");
  ASSERT_STREQ(s.c_str(), "");
  ASSERT_EQ(s.data(), s.c_str());
  ASSERT_EQ(s.capacity(), size_t(0));
}

TEST(LlvmLibcStringTest, InitializeCString) {
  const char *const str = "abc";
  const string s(str);
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_FALSE(s.empty());
  ASSERT_NE(s.data(), &str[0]);
  ASSERT_EQ(s[0], 'a');
  ASSERT_EQ(s[1], 'b');
  ASSERT_EQ(s[2], 'c');
  ASSERT_EQ(s.front(), 'a');
  ASSERT_EQ(s.back(), 'c');
  ASSERT_EQ(s.data(), s.c_str());
}

TEST(LlvmLibcStringTest, ToCString) {
  const char *const str = "abc";
  string s(str);
  const char *cstr = s.c_str();
  ASSERT_EQ(s.size(), size_t(3));
  ASSERT_STREQ(str, cstr);
}

TEST(LlvmLibcStringTest, ToStringView) {
  const char *const str = "abc";
  string s(str);
  string_view view = s;
  ASSERT_EQ(view, string_view(str));
}

TEST(LlvmLibcStringTest, InitializeCStringWithSize) {
  const char *const str = "abc";
  const string s(str, 2);
  ASSERT_EQ(s.size(), size_t(2));
  ASSERT_EQ(s[0], 'a');
  ASSERT_EQ(s[1], 'b');
  ASSERT_EQ(s.front(), 'a');
  ASSERT_EQ(s.back(), 'b');
}

TEST(LlvmLibcStringTest, InitializeStringView) {
  const string_view str = "ab";
  const string s(str);
  ASSERT_EQ(s.size(), size_t(2));
  ASSERT_EQ(s[0], 'a');
  ASSERT_EQ(s[1], 'b');
  ASSERT_EQ(s.front(), 'a');
  ASSERT_EQ(s.back(), 'b');
}

TEST(LlvmLibcStringTest, InitializeRepeatedChar) {
  const string s(4, '1');
  ASSERT_EQ(string_view(s), string_view("1111"));
}

TEST(LlvmLibcStringTest, InitializeZeroChar) {
  const string s(0, '1');
  ASSERT_TRUE(s.empty());
}

TEST(LlvmLibcStringTest, CopyConstruct) {
  const char *const str = "abc";
  string a(str);
  string b(a);
  // Same content
  ASSERT_STREQ(a.c_str(), str);
  ASSERT_STREQ(b.c_str(), str);
  // Different pointers
  ASSERT_NE(a.data(), b.data());
}

string &&move(string &value) { return static_cast<string &&>(value); }

TEST(LlvmLibcStringTest, CopyAssign) {
  const char *const str = "abc";
  string a(str);
  string b;
  b = a;
  // Same content
  ASSERT_STREQ(a.c_str(), str);
  ASSERT_STREQ(b.c_str(), str);
  // Different pointers
  ASSERT_NE(a.data(), b.data());
}

TEST(LlvmLibcStringTest, MoveConstruct) {
  const char *const str = "abc";
  string a(str);
  string b(move(a));
  ASSERT_STREQ(b.c_str(), str);
  ASSERT_STREQ(a.c_str(), "");
}

TEST(LlvmLibcStringTest, MoveAssign) {
  const char *const str = "abc";
  string a(str);
  string b;
  b = move(a);
  ASSERT_STREQ(b.c_str(), str);
  ASSERT_STREQ(a.c_str(), "");
}

TEST(LlvmLibcStringTest, StringViewAssign) {
  const string_view str = "ab";
  string s;
  s = str;
  ASSERT_EQ(s.size(), size_t(2));
  ASSERT_EQ(s[0], 'a');
  ASSERT_EQ(s[1], 'b');
  ASSERT_EQ(s.front(), 'a');
  ASSERT_EQ(s.back(), 'b');
}

TEST(LlvmLibcStringTest, CharacterAssign) {
  string s("abc");
  s = 'a';
  ASSERT_EQ(s.size(), size_t(1));
  ASSERT_STREQ(s.c_str(), "a");
}

TEST(LlvmLibcStringTest, Concat) {
  const char *const str = "abc";
  string a(str);
  string b;
  b += a;
  ASSERT_STREQ(b.c_str(), "abc");
  b += a;
  ASSERT_STREQ(b.c_str(), "abcabc");
}

TEST(LlvmLibcStringTest, AddChar) {
  string a;
  a += 'a';
  ASSERT_STREQ(a.c_str(), "a");
  a += 'b';
  ASSERT_STREQ(a.c_str(), "ab");
}

TEST(LlvmLibcStringTest, ResizeCapacityAndNullTermination) {
  string a;
  // Empty
  ASSERT_EQ(a.capacity(), size_t(0));
  ASSERT_EQ(a.data()[0], '\0');
  // Still empty
  a.resize(0);
  ASSERT_EQ(a.capacity(), size_t(0));
  ASSERT_EQ(a.data()[0], '\0');
  // One char
  a.resize(1);
  ASSERT_EQ(a.size(), size_t(1));
  ASSERT_GE(a.capacity(), size_t(2));
  ASSERT_EQ(a.data()[1], '\0');
  // Clear
  a.resize(0);
  ASSERT_EQ(a.size(), size_t(0));
  ASSERT_GE(a.capacity(), size_t(2));
  ASSERT_EQ(a.data()[0], '\0');
  // Resize and check zero initialized
  a.resize(10);
  ASSERT_EQ(a.size(), size_t(10));
  ASSERT_GE(a.capacity(), size_t(10));
  for (size_t i = 0; i < 10; ++i)
    ASSERT_EQ(a[i], '\0');
}

TEST(LlvmLibcStringTest, ResizeWithCapacityPlus1) {
  string a;
  a.resize(32);

  size_t previous_capacity = a.capacity();
  a.resize(previous_capacity);
  ASSERT_GT(a.capacity(), previous_capacity);
}

TEST(LlvmLibcStringTest, ConcatWithCString) {
  ASSERT_STREQ((string("a") + string("b")).c_str(), "ab");
  ASSERT_STREQ((string("a") + "b").c_str(), "ab");
  ASSERT_STREQ(("a" + string("b")).c_str(), "ab");
}

TEST(LlvmLibcStringTest, ReleaseCStrResetsString) {
  string s("abc");

  char *cstr = s.release_c_str();
  ASSERT_STREQ(cstr, "abc");
  ASSERT_STREQ(s.c_str(), "");
  free(cstr);
}

TEST(LlvmLibcStringTest, ReleaseCStrOnDefaultString) {
  string s;

  char *cstr = s.release_c_str();
  ASSERT_STREQ(cstr, "");
  ASSERT_STREQ(s.c_str(), "");
  ASSERT_NE(cstr, s.data()); // Different pointers.
  free(cstr);
}

TEST(LlvmLibcStringTest, Comparison) {
  // Here we simply check that comparison of string and string_view have the
  // same semantic.
  struct CStringPair {
    const char *const a;
    const char *const b;
  } kTestPairs[] = {{"a", "b"}, {"", "xyz"}};
  for (const auto [pa, pb] : kTestPairs) {
    const string sa(pa);
    const string sb(pb);
    const string_view sva(pa);
    const string_view svb(pb);
    ASSERT_EQ(sa == sb, sva == svb);
    ASSERT_EQ(sa != sb, sva != svb);
    ASSERT_EQ(sa >= sb, sva >= svb);
    ASSERT_EQ(sa <= sb, sva <= svb);
    ASSERT_EQ(sa < sb, sva < svb);
    ASSERT_EQ(sa > sb, sva > svb);
  }
}

TEST(LlvmLibcStringTest, ToString) {
  struct CStringPair {
    const int value;
    const string str;
  } kTestPairs[] = {{123, "123"}, {0, "0"}, {-321, "-321"}};
  for (const auto &[value, str] : kTestPairs) {
    ASSERT_EQ(to_string((int)(value)), str);
    ASSERT_EQ(to_string((long)(value)), str);
    ASSERT_EQ(to_string((long long)(value)), str);
    if (value >= 0) {
      ASSERT_EQ(to_string((unsigned int)(value)), str);
      ASSERT_EQ(to_string((unsigned long)(value)), str);
      ASSERT_EQ(to_string((unsigned long long)(value)), str);
    }
  }
}

TEST(LlvmLibcStringTest, SelfAssignTest) {
  string s("abcdefg");
  s = string_view(s).substr(1);
  ASSERT_STREQ(s.c_str(), "bcdefg");
}

TEST(LlvmLibcStringTest, SelfAssignAtCapacityTest) {
  string s("aaa");

  // Append until the string is at its capacity
  // to exercise self-appending past capacity.
  while (s.size() + 1 < s.capacity())
    s += 'a';
  ASSERT_EQ(s.capacity(), s.size() + 1);

  string longer_string(s.size() + 1, 'b');

  // Force a resize,
  s = string_view(longer_string);
  ASSERT_EQ(s, longer_string);
}

TEST(LlvmLibcStringTest, SelfAppendAtCapacityTest) {
  string s("aaa");

  // Append until the string is at its capacity
  // to exercise self-appending past capacity.
  while (s.size() + 1 < s.capacity())
    s += 'a';
  ASSERT_EQ(s.capacity(), s.size() + 1);

  string_view view = string_view(s).substr(0, 3);
  size_t expected_size = s.size() + view.size();

  s += view;

  string expected(expected_size, 'a');
  ASSERT_EQ(s, expected);
}
