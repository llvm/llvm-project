//===-- Unittests for StringView ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/StringView.h"
#include "utils/UnitTest/Test.h"

using __llvm_libc::cpp::StringView;

TEST(LlvmLibcStringViewTest, InitializeCheck) {
  StringView v;
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = StringView("");
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = StringView(nullptr);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = StringView(nullptr, 10);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = StringView("abc", 0);
  ASSERT_EQ(v.size(), size_t(0));
  ASSERT_TRUE(v.data() == nullptr);

  v = StringView("123456789");
  ASSERT_EQ(v.size(), size_t(9));
}

TEST(LlvmLibcStringViewTest, Equals) {
  StringView v("abc");
  ASSERT_TRUE(v.equals(StringView("abc")));
  ASSERT_FALSE(v.equals(StringView()));
  ASSERT_FALSE(v.equals(StringView("")));
  ASSERT_FALSE(v.equals(StringView("123")));
  ASSERT_FALSE(v.equals(StringView("abd")));
  ASSERT_FALSE(v.equals(StringView("aaa")));
  ASSERT_FALSE(v.equals(StringView("abcde")));
}

TEST(LlvmLibcStringViewTest, startsWith) {
  StringView v("abc");
  ASSERT_TRUE(v.starts_with('a'));
  ASSERT_TRUE(v.starts_with(StringView("a")));
  ASSERT_TRUE(v.starts_with(StringView("ab")));
  ASSERT_TRUE(v.starts_with(StringView("abc")));
  ASSERT_TRUE(v.starts_with(StringView()));
  ASSERT_TRUE(v.starts_with(StringView("")));
  ASSERT_FALSE(v.starts_with('1'));
  ASSERT_FALSE(v.starts_with(StringView("123")));
  ASSERT_FALSE(v.starts_with(StringView("abd")));
  ASSERT_FALSE(v.starts_with(StringView("aaa")));
  ASSERT_FALSE(v.starts_with(StringView("abcde")));
}

TEST(LlvmLibcStringViewTest, endsWith) {
  StringView v("abc");
  ASSERT_TRUE(v.ends_with('c'));
  ASSERT_TRUE(v.ends_with(StringView("c")));
  ASSERT_TRUE(v.ends_with(StringView("bc")));
  ASSERT_TRUE(v.ends_with(StringView("abc")));
  ASSERT_TRUE(v.ends_with(StringView()));
  ASSERT_TRUE(v.ends_with(StringView("")));
  ASSERT_FALSE(v.ends_with('1'));
  ASSERT_FALSE(v.ends_with(StringView("123")));
  ASSERT_FALSE(v.ends_with(StringView("abd")));
  ASSERT_FALSE(v.ends_with(StringView("aaa")));
  ASSERT_FALSE(v.ends_with(StringView("abcde")));
}

TEST(LlvmLibcStringViewTest, RemovePrefix) {
  StringView a("123456789");
  a.remove_prefix(0);
  ASSERT_EQ(a.size(), size_t(9));
  ASSERT_TRUE(a.equals(StringView("123456789")));

  StringView b("123456789");
  b.remove_prefix(4);
  ASSERT_EQ(b.size(), size_t(5));
  ASSERT_TRUE(b.equals(StringView("56789")));

  StringView c("123456789");
  c.remove_prefix(9);
  ASSERT_EQ(c.size(), size_t(0));
}

TEST(LlvmLibcStringViewTest, RemoveSuffix) {
  StringView a("123456789");
  a.remove_suffix(0);
  ASSERT_EQ(a.size(), size_t(9));
  ASSERT_TRUE(a.equals(StringView("123456789")));

  StringView b("123456789");
  b.remove_suffix(4);
  ASSERT_EQ(b.size(), size_t(5));
  ASSERT_TRUE(b.equals(StringView("12345")));

  StringView c("123456789");
  c.remove_suffix(9);
  ASSERT_EQ(c.size(), size_t(0));
}

TEST(LlvmLibcStringViewTest, TrimSingleChar) {
  StringView v("     123456789   ");
  auto t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(9));
  ASSERT_TRUE(t.equals(StringView("123456789")));

  v = StringView("====12345==");
  t = v.trim(' ');
  ASSERT_EQ(v.size(), size_t(11));
  ASSERT_TRUE(t.equals(StringView("====12345==")));

  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(StringView("12345")));

  v = StringView("12345===");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(StringView("12345")));

  v = StringView("===========12345");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(5));
  ASSERT_TRUE(t.equals(StringView("12345")));

  v = StringView("============");
  t = v.trim('=');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);

  v = StringView();
  t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);

  v = StringView("");
  t = v.trim(' ');
  ASSERT_EQ(t.size(), size_t(0));
  ASSERT_TRUE(t.data() == nullptr);
}

TEST(LlvmLibcStringViewTest, Observer) {
  StringView ABC("abc");
  ASSERT_EQ(ABC.size(), size_t(3));
  ASSERT_FALSE(ABC.empty());
  ASSERT_EQ(ABC.front(), 'a');
  ASSERT_EQ(ABC.back(), 'c');
}

bool isDigit(char c) { return c >= '0' && c <= '9'; }

TEST(LlvmLibcStringViewTest, Transform) {
  ASSERT_TRUE(StringView("123abc").drop_back(3).equals("123"));
  ASSERT_TRUE(StringView("123abc").drop_front(3).equals("abc"));
  ASSERT_TRUE(StringView("123abc").take_back(3).equals("abc"));
  ASSERT_TRUE(StringView("123abc").take_front(3).equals("123"));

  ASSERT_TRUE(StringView("123abc").take_while(&isDigit).equals("123"));
  ASSERT_TRUE(StringView("abc123").take_until(&isDigit).equals("abc"));
  ASSERT_TRUE(StringView("123abc").drop_while(&isDigit).equals("abc"));
  ASSERT_TRUE(StringView("abc123").drop_until(&isDigit).equals("123"));
}

TEST(LlvmLibcStringViewTest, ConsumeFront) {
  StringView Tmp("abc");
  ASSERT_FALSE(Tmp.consume_front("###"));
  ASSERT_TRUE(Tmp.consume_front("ab"));
  ASSERT_TRUE(Tmp.equals("c"));
}

TEST(LlvmLibcStringViewTest, ConsumeBack) {
  StringView Tmp("abc");
  ASSERT_FALSE(Tmp.consume_back("###"));
  ASSERT_TRUE(Tmp.consume_back("bc"));
  ASSERT_TRUE(Tmp.equals("a"));
}

TEST(LlvmLibcStringViewTest, FindFirstOf) {
  StringView Tmp("abca");
  ASSERT_TRUE(Tmp.find_first_of('a') == 0);
  ASSERT_TRUE(Tmp.find_first_of('d') == StringView::npos);
  ASSERT_TRUE(Tmp.find_first_of('b') == 1);
  ASSERT_TRUE(Tmp.find_first_of('a', 0) == 0);
  ASSERT_TRUE(Tmp.find_first_of('b', 1) == 1);
  ASSERT_TRUE(Tmp.find_first_of('a', 1) == 3);
  ASSERT_TRUE(Tmp.find_first_of('a', 42) == StringView::npos);
  ASSERT_FALSE(Tmp.find_first_of('c') == 1);
  ASSERT_FALSE(Tmp.find_first_of('c', 0) == 1);
  ASSERT_FALSE(Tmp.find_first_of('c', 1) == 1);
}
