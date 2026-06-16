//===-- Unittests for string_view -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/UnitTest/Test.h"

using LIBC_NAMESPACE::cpp::string_view;
using LIBC_NAMESPACE::cpp::wstring_view;

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

  v = string_view("123456789", 5);
  ASSERT_EQ(v.size(), size_t(5));

  wstring_view WideStr;
  ASSERT_EQ(WideStr.size(), size_t(0));
  ASSERT_TRUE(WideStr.data() == nullptr);

  WideStr = wstring_view(L"");
  ASSERT_EQ(WideStr.size(), size_t(0));
  ASSERT_TRUE(WideStr.data() != nullptr);

  WideStr = wstring_view(L"abc", 0);
  ASSERT_EQ(WideStr.size(), size_t(0));
  ASSERT_TRUE(WideStr.data() != nullptr);

  WideStr = wstring_view(L"123456789");
  ASSERT_EQ(WideStr.size(), size_t(9));

  WideStr = wstring_view(L"123456789", 5);
  ASSERT_EQ(WideStr.size(), size_t(5));
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

  wstring_view WideStr(L"abc");
  ASSERT_EQ(WideStr, wstring_view(L"abc"));
  ASSERT_NE(WideStr, wstring_view());
  ASSERT_NE(WideStr, wstring_view(L""));
  ASSERT_NE(WideStr, wstring_view(L"123"));
  ASSERT_NE(WideStr, wstring_view(L"abd"));
  ASSERT_NE(WideStr, wstring_view(L"aaa"));
  ASSERT_NE(WideStr, wstring_view(L"abcde"));
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

  wstring_view WideStr(L"abc");
  ASSERT_TRUE(WideStr.starts_with(L'a'));
  ASSERT_TRUE(WideStr.starts_with(wstring_view(L"a")));
  ASSERT_TRUE(WideStr.starts_with(wstring_view(L"ab")));
  ASSERT_TRUE(WideStr.starts_with(wstring_view(L"abc")));
  ASSERT_TRUE(WideStr.starts_with(wstring_view()));
  ASSERT_TRUE(WideStr.starts_with(wstring_view(L"")));
  ASSERT_FALSE(WideStr.starts_with(L'1'));
  ASSERT_FALSE(WideStr.starts_with(wstring_view(L"123")));
  ASSERT_FALSE(WideStr.starts_with(wstring_view(L"abd")));
  ASSERT_FALSE(WideStr.starts_with(wstring_view(L"aaa")));
  ASSERT_FALSE(WideStr.starts_with(wstring_view(L"abcde")));
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

  wstring_view WideStr(L"abc");
  ASSERT_TRUE(WideStr.ends_with(L'c'));
  ASSERT_TRUE(WideStr.ends_with(wstring_view(L"c")));
  ASSERT_TRUE(WideStr.ends_with(wstring_view(L"bc")));
  ASSERT_TRUE(WideStr.ends_with(wstring_view(L"abc")));
  ASSERT_TRUE(WideStr.ends_with(wstring_view()));
  ASSERT_TRUE(WideStr.ends_with(wstring_view(L"")));
  ASSERT_FALSE(WideStr.ends_with(L'1'));
  ASSERT_FALSE(WideStr.ends_with(wstring_view(L"123")));
  ASSERT_FALSE(WideStr.ends_with(wstring_view(L"abd")));
  ASSERT_FALSE(WideStr.ends_with(wstring_view(L"aaa")));
  ASSERT_FALSE(WideStr.ends_with(wstring_view(L"abcde")));
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

  wstring_view WideStrA(L"123456789");
  WideStrA.remove_prefix(0);
  ASSERT_EQ(WideStrA.size(), size_t(9));
  ASSERT_TRUE(WideStrA == L"123456789");

  wstring_view WideStrB(L"123456789");
  WideStrB.remove_prefix(4);
  ASSERT_EQ(WideStrB.size(), size_t(5));
  ASSERT_TRUE(WideStrB == L"56789");

  wstring_view WideStrC(L"123456789");
  WideStrC.remove_prefix(9);
  ASSERT_EQ(WideStrC.size(), size_t(0));
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

  wstring_view WideStrA(L"123456789");
  WideStrA.remove_suffix(0);
  ASSERT_EQ(WideStrA.size(), size_t(9));
  ASSERT_TRUE(WideStrA == L"123456789");

  wstring_view WideStrB(L"123456789");
  WideStrB.remove_suffix(4);
  ASSERT_EQ(WideStrB.size(), size_t(5));
  ASSERT_TRUE(WideStrB == L"12345");

  wstring_view WideStrC(L"123456789");
  WideStrC.remove_suffix(9);
  ASSERT_EQ(WideStrC.size(), size_t(0));
}

TEST(LlvmLibcStringViewTest, Observer) {
  string_view ABC("abc");
  ASSERT_EQ(ABC.size(), size_t(3));
  ASSERT_FALSE(ABC.empty());
  ASSERT_EQ(ABC.front(), 'a');
  ASSERT_EQ(ABC.back(), 'c');

  wstring_view WideStr(L"abc");
  ASSERT_EQ(WideStr.size(), size_t(3));
  ASSERT_FALSE(WideStr.empty());
  ASSERT_EQ(WideStr.front(), L'a');
  ASSERT_EQ(WideStr.back(), L'c');
}

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

  wstring_view WideStr(L"abca");
  ASSERT_TRUE(WideStr.find_first_of(L'a') == 0);
  ASSERT_TRUE(WideStr.find_first_of(L'd') == wstring_view::npos);
  ASSERT_TRUE(WideStr.find_first_of(L'b') == 1);
  ASSERT_TRUE(WideStr.find_first_of(L'a', 0) == 0);
  ASSERT_TRUE(WideStr.find_first_of(L'b', 1) == 1);
  ASSERT_TRUE(WideStr.find_first_of(L'a', 1) == 3);
  ASSERT_TRUE(WideStr.find_first_of(L'a', 42) == wstring_view::npos);
  ASSERT_FALSE(WideStr.find_first_of(L'c') == 1);
  ASSERT_FALSE(WideStr.find_first_of(L'c', 0) == 1);
  ASSERT_FALSE(WideStr.find_first_of(L'c', 1) == 1);
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

  wstring_view WideStr(L"abada");
  ASSERT_EQ(WideStr.find_last_of(L'a'), size_t(4));
  ASSERT_EQ(WideStr.find_last_of(L'a', 123), size_t(4));
  ASSERT_EQ(WideStr.find_last_of(L'a', 5), size_t(4));
  ASSERT_EQ(WideStr.find_last_of(L'a', 4), size_t(4));
  ASSERT_EQ(WideStr.find_last_of(L'a', 3), size_t(2));
  ASSERT_EQ(WideStr.find_last_of(L'a', 2), size_t(2));
  ASSERT_EQ(WideStr.find_last_of(L'a', 1), size_t(0));
  ASSERT_EQ(WideStr.find_last_of(L'a', 0), size_t(0));

  ASSERT_EQ(WideStr.find_last_of(L'b'), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 123), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 5), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 4), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 3), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 2), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 1), size_t(1));
  ASSERT_EQ(WideStr.find_last_of(L'b', 0), wstring_view::npos);

  ASSERT_EQ(WideStr.find_last_of(L'd'), size_t(3));
  ASSERT_EQ(WideStr.find_last_of(L'd', 123), size_t(3));
  ASSERT_EQ(WideStr.find_last_of(L'd', 5), size_t(3));
  ASSERT_EQ(WideStr.find_last_of(L'd', 4), size_t(3));
  ASSERT_EQ(WideStr.find_last_of(L'd', 3), size_t(3));
  ASSERT_EQ(WideStr.find_last_of(L'd', 2), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'd', 1), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'd', 0), wstring_view::npos);

  ASSERT_EQ(WideStr.find_last_of(L'e'), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 123), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 5), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 4), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 3), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 2), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 1), wstring_view::npos);
  ASSERT_EQ(WideStr.find_last_of(L'e', 0), wstring_view::npos);

  wstring_view WideEmpty;
  ASSERT_EQ(WideEmpty.find_last_of(L'a'), wstring_view::npos);
  ASSERT_EQ(WideEmpty.find_last_of(L'a', 0), wstring_view::npos);
  ASSERT_EQ(WideEmpty.find_last_of(L'a', 123), wstring_view::npos);

  WideEmpty = L"";
  ASSERT_EQ(WideEmpty.find_last_of(L'a'), wstring_view::npos);
  ASSERT_EQ(WideEmpty.find_last_of(L'a', 0), wstring_view::npos);
  ASSERT_EQ(WideEmpty.find_last_of(L'a', 123), wstring_view::npos);
}

TEST(LlvmLibcStringViewTest, FindFirstNotOf) {
  string_view Tmp("abada");

  EXPECT_EQ(Tmp.find_first_not_of('a'), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of('a', 123), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('a', 5), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('a', 4), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('a', 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of('a', 2), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of('a', 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of('a', 0), size_t(1));

  EXPECT_EQ(Tmp.find_first_not_of('b'), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of('b', 123), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('b', 5), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('b', 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of('b', 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of('b', 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of('b', 1), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of('b', 0), size_t(0));

  EXPECT_EQ(Tmp.find_first_not_of('d'), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of('d', 123), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('d', 5), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('d', 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of('d', 3), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of('d', 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of('d', 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of('d', 0), size_t(0));

  EXPECT_EQ(Tmp.find_first_not_of('e'), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of('e', 123), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('e', 5), string_view::npos);
  EXPECT_EQ(Tmp.find_first_not_of('e', 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of('e', 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of('e', 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of('e', 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of('e', 0), size_t(0));

  string_view Empty;
  EXPECT_EQ(Empty.find_first_not_of('a'), string_view::npos);
  EXPECT_EQ(Empty.find_first_not_of('a', 0), string_view::npos);
  EXPECT_EQ(Empty.find_first_not_of('a', 123), string_view::npos);

  string_view Empty1("");
  EXPECT_EQ(Empty1.find_first_not_of('a'), string_view::npos);
  EXPECT_EQ(Empty1.find_first_not_of('a', 0), string_view::npos);
  EXPECT_EQ(Empty1.find_first_not_of('a', 123), string_view::npos);

  string_view Full("aaaaaaa");
  EXPECT_EQ(Full.find_first_not_of('a'), string_view::npos);
  EXPECT_EQ(Full.find_first_not_of('a', 0), string_view::npos);
  EXPECT_EQ(Full.find_first_not_of('a', 123), string_view::npos);

  EXPECT_EQ(Full.find_first_not_of('b'), size_t(0));
  EXPECT_EQ(Full.find_first_not_of('b', 0), size_t(0));
  EXPECT_EQ(Full.find_first_not_of('b', 123), string_view::npos);

  wstring_view WideStr(L"abada");
  EXPECT_EQ(WideStr.find_first_not_of(L'a'), size_t(1));
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 123), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 5), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 4), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 3), size_t(3));
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 2), size_t(3));
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 1), size_t(1));
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 0), size_t(1));

  EXPECT_EQ(WideStr.find_first_not_of(L'b'), size_t(0));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 123), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 5), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 4), size_t(4));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 3), size_t(3));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 2), size_t(2));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 1), size_t(2));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 0), size_t(0));

  EXPECT_EQ(WideStr.find_first_not_of(L'd'), size_t(0));
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 123), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 5), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 4), size_t(4));
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 3), size_t(4));
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 2), size_t(2));
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 1), size_t(1));
  EXPECT_EQ(WideStr.find_first_not_of(L'd', 0), size_t(0));

  EXPECT_EQ(WideStr.find_first_not_of(L'e'), size_t(0));
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 123), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 5), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 4), size_t(4));
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 3), size_t(3));
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 2), size_t(2));
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 1), size_t(1));
  EXPECT_EQ(WideStr.find_first_not_of(L'e', 0), size_t(0));

  wstring_view WideEmpty;
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a'), wstring_view::npos);
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a', 0), wstring_view::npos);
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a', 123), wstring_view::npos);

  WideEmpty = L"";
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a'), wstring_view::npos);
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a', 0), wstring_view::npos);
  EXPECT_EQ(WideEmpty.find_first_not_of(L'a', 123), wstring_view::npos);

  WideStr = L"aaaaaaa";
  EXPECT_EQ(WideStr.find_first_not_of(L'a'), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 0), wstring_view::npos);
  EXPECT_EQ(WideStr.find_first_not_of(L'a', 123), wstring_view::npos);

  EXPECT_EQ(WideStr.find_first_not_of(L'b'), size_t(0));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 0), size_t(0));
  EXPECT_EQ(WideStr.find_first_not_of(L'b', 123), wstring_view::npos);
}

TEST(LlvmLibcStringViewTest, Contains) {
  string_view Empty;
  EXPECT_FALSE(Empty.contains('a'));
  EXPECT_FALSE(Empty.contains('g'));
  EXPECT_FALSE(Empty.contains('q'));

  wstring_view WideEmpty;
  EXPECT_FALSE(WideEmpty.contains('a'));
  EXPECT_FALSE(WideEmpty.contains('g'));
  EXPECT_FALSE(WideEmpty.contains('q'));

  string_view Str = "abada";
  EXPECT_TRUE(Str.contains('a'));
  EXPECT_TRUE(Str.contains('b'));
  EXPECT_FALSE(Str.contains('c'));
  EXPECT_TRUE(Str.contains('d'));
  EXPECT_FALSE(Str.contains('e'));

  wstring_view WideStr = L"abada";
  EXPECT_TRUE(WideStr.contains(L'a'));
  EXPECT_TRUE(WideStr.contains(L'b'));
  EXPECT_FALSE(WideStr.contains(L'c'));
  EXPECT_TRUE(WideStr.contains(L'd'));
  EXPECT_FALSE(WideStr.contains(L'e'));
}

TEST(LlvmLibcStringViewTest, Substr) {
  string_view Str = "abada";
  EXPECT_EQ(Str.substr(0), string_view("abada"));
  EXPECT_EQ(Str.substr(1), string_view("bada"));
  EXPECT_EQ(Str.substr(3), string_view("da"));
  EXPECT_EQ(Str.substr(5), string_view(""));
  EXPECT_EQ(Str.substr(1, 3), string_view("bad"));
  EXPECT_EQ(Str.substr(3, 1), string_view("d"));

  wstring_view WideStr = L"abada";
  EXPECT_EQ(WideStr.substr(0), wstring_view(L"abada"));
  EXPECT_EQ(WideStr.substr(1), wstring_view(L"bada"));
  EXPECT_EQ(WideStr.substr(3), wstring_view(L"da"));
  EXPECT_EQ(WideStr.substr(5), wstring_view(L""));
  EXPECT_EQ(WideStr.substr(1, 3), wstring_view(L"bad"));
  EXPECT_EQ(WideStr.substr(3, 1), wstring_view(L"d"));
}

TEST(LlvmLibcStringViewTest, WideCharacterComparison) {
  // Check that wide character comparison is lexicographic by character and not
  // equivalent to memcmp, which would be incorrect on little endian.
  char BytesA[] = {1, 2, 3, 4};
  char BytesB[] = {4, 3, 2, 1};
  char32_t CharA;
  char32_t CharB;
  LIBC_NAMESPACE::inline_memcpy(&CharA, BytesA, sizeof(CharA));
  LIBC_NAMESPACE::inline_memcpy(&CharB, BytesB, sizeof(CharB));

  LIBC_NAMESPACE::cpp::basic_string_view<char32_t> StringViewA(&CharA, 1);
  LIBC_NAMESPACE::cpp::basic_string_view<char32_t> StringViewB(&CharB, 1);

  EXPECT_EQ(StringViewA < StringViewB, CharA < CharB);
}
