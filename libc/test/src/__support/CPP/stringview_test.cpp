//===-- Unittests for string_view
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::string_view;

TEST(LlvmLibcStringViewTest, InitializeCheck) {
  string_view v;
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = string_view("");
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() != nullptr);

  v = string_view("abc", 0);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() != nullptr);

  v = string_view("123456789");
  ASSERT_EQ(v.size(), size_t(9));
}

TEST(LlvmLibcStringViewTest, Equals) {
  string_view v("abc");
  ASSERT_EQ(v, string_view("abc"));
  ASSERT_NE(v, string_view());
  ASSERT_NE(v, string_view(""));
  ASSERT_NE(v, string_view("123"));
  ASSERT_NE(v, string_view("abd"));
  ASSERT_NE(v, string_view("aaa"));
  ASSERT_NE(v, string_view("abcde"));
}

TEST(LlvmLibcStringViewTest, startsWith) {
  string_view v("abc");
  ASSERT_TRUE(v.starts_with('a'));
  ASSERT_TRUE(v.starts_with(string_view("a")));
  ASSERT_TRUE(v.starts_with(string_view("ab")));
  ASSERT_TRUE(v.starts_with(string_view("abc")));
  ASSERT_TRUE(v.starts_with(string_view()));
  ASSERT_TRUE(v.starts_with(string_view("")));
  ASSERT_FALSE(v.starts_with('1'));
  ASSERT_FALSE(v.starts_with(string_view("123")));
  ASSERT_FALSE(v.starts_with(string_view("abd")));
  ASSERT_FALSE(v.starts_with(string_view("aaa")));
  ASSERT_FALSE(v.starts_with(string_view("abcde")));
}

TEST(LlvmLibcStringViewTest, endsWith) {
  string_view v("abc");
  ASSERT_TRUE(v.ends_with('c'));
  ASSERT_TRUE(v.ends_with(string_view("c")));
  ASSERT_TRUE(v.ends_with(string_view("bc")));
  ASSERT_TRUE(v.ends_with(string_view("abc")));
  ASSERT_TRUE(v.ends_with(string_view()));
  ASSERT_TRUE(v.ends_with(string_view("")));
  ASSERT_FALSE(v.ends_with('1'));
  ASSERT_FALSE(v.ends_with(string_view("123")));
  ASSERT_FALSE(v.ends_with(string_view("abd")));
  ASSERT_FALSE(v.ends_with(string_view("aaa")));
  ASSERT_FALSE(v.ends_with(string_view("abcde")));
}

TEST(LlvmLibcStringViewTest, RemovePrefix) {
  string_view a("123456789");
  a.remove_prefix(0);
  ASSERT_EQ(a.size(), size_t(9));
  ASSERT_TRUE(a == "123456789");

  string_view b("123456789");
  b.remove_prefix(4);
  ASSERT_EQ(b.size(), size_t(5));
  ASSERT_TRUE(b == "56789");

  string_view c("123456789");
  c.remove_prefix(9);
  ASSERT_EQ(c.size(), size_t(0));
}

TEST(LlvmLibcStringViewTest, RemoveSuffix) {
  string_view a("123456789");
  a.remove_suffix(0);
  ASSERT_EQ(a.size(), size_t(9));
  ASSERT_TRUE(a == "123456789");

  string_view b("123456789");
  b.remove_suffix(4);
  ASSERT_EQ(b.size(), size_t(5));
  ASSERT_TRUE(b == "12345");

  string_view c("123456789");
  c.remove_suffix(9);
  ASSERT_EQ(c.size(), size_t(0));
}

TEST(LlvmLibcStringViewTest, Observer) {
  string_view ABC("abc");
  ASSERT_EQ(ABC.size(), size_t(3));
  ASSERT_FALSE(ABC.empty());
  ASSERT_EQ(ABC.front(), 'a');
  ASSERT_EQ(ABC.back(), 'c');
}

bool isDigit(char c) { return c >= '0' && c <= '9'; }

TEST(LlvmLibcStringViewTest, FindFirstOf) {
  string_view Tmp("abca");
  ASSERT_TRUE(Tmp.find_first_of('a') == 0);
  ASSERT_TRUE(Tmp.find_first_of('d') == string_view::npos);
  ASSERT_TRUE(Tmp.find_first_of('b') == 1);
  ASSERT_TRUE(Tmp.find_first_of('a', 0) == 0);
  ASSERT_TRUE(Tmp.find_first_of('b', 1) == 1);
  ASSERT_TRUE(Tmp.find_first_of('a', 1) == 3);
  ASSERT_TRUE(Tmp.find_first_of('a', 42) == string_view::npos);
  ASSERT_FALSE(Tmp.find_first_of('c') == 1);
  ASSERT_FALSE(Tmp.find_first_of('c', 0) == 1);
  ASSERT_FALSE(Tmp.find_first_of('c', 1) == 1);
}

TEST(LlvmLibcStringViewTest, FindLastOf) {
  string_view Tmp("abada");

  ASSERT_EQ(Tmp.find_last_of('a'), size_t(4));
  ASSERT_EQ(Tmp.find_last_of('a', 123), size_t(4));
  ASSERT_EQ(Tmp.find_last_of('a', 5), size_t(4));
  ASSERT_EQ(Tmp.find_last_of('a', 4), size_t(4));
  ASSERT_EQ(Tmp.find_last_of('a', 3), size_t(2));
  ASSERT_EQ(Tmp.find_last_of('a', 2), size_t(2));
  ASSERT_EQ(Tmp.find_last_of('a', 1), size_t(0));
  ASSERT_EQ(Tmp.find_last_of('a', 0), size_t(0));

  ASSERT_EQ(Tmp.find_last_of('b'), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 123), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 5), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 4), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 3), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 2), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 1), size_t(1));
  ASSERT_EQ(Tmp.find_last_of('b', 0), string_view::npos);

  ASSERT_EQ(Tmp.find_last_of('d'), size_t(3));
  ASSERT_EQ(Tmp.find_last_of('d', 123), size_t(3));
  ASSERT_EQ(Tmp.find_last_of('d', 5), size_t(3));
  ASSERT_EQ(Tmp.find_last_of('d', 4), size_t(3));
  ASSERT_EQ(Tmp.find_last_of('d', 3), size_t(3));
  ASSERT_EQ(Tmp.find_last_of('d', 2), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('d', 1), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('d', 0), string_view::npos);

  ASSERT_EQ(Tmp.find_last_of('e'), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 123), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 5), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 4), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 3), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 2), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 1), string_view::npos);
  ASSERT_EQ(Tmp.find_last_of('e', 0), string_view::npos);

  string_view Empty;
  ASSERT_EQ(Empty.find_last_of('a'), string_view::npos);
  ASSERT_EQ(Empty.find_last_of('a', 0), string_view::npos);
  ASSERT_EQ(Empty.find_last_of('a', 123), string_view::npos);

  string_view Empty1("");
  ASSERT_EQ(Empty1.find_last_of('a'), string_view::npos);
  ASSERT_EQ(Empty1.find_last_of('a', 0), string_view::npos);
  ASSERT_EQ(Empty1.find_last_of('a', 123), string_view::npos);
}
