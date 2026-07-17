//===-- Unittests for string_view -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/string/memory_utils/inline_memcpy.h"
#include "test/UnitTest/Test.h"

using TestCharTypes = LIBC_NAMESPACE::testing::TypeList<char, wchar_t>;

template <typename CharT>
const CharT *chooseLiteral(const char *CharStr, const wchar_t *WCharStr) {
  if constexpr (LIBC_NAMESPACE::cpp::is_same_v<CharT, char>)
    return CharStr;
  else {
    static_assert(LIBC_NAMESPACE::cpp::is_same_v<CharT, wchar_t>);
    return WCharStr;
  }
}

template <typename CharT>
CharT chooseLiteral(char CharValue, wchar_t WCharValue) {
  if constexpr (LIBC_NAMESPACE::cpp::is_same_v<CharT, char>)
    return CharValue;
  else {
    static_assert(LIBC_NAMESPACE::cpp::is_same_v<CharT, wchar_t>);
    return WCharValue;
  }
}

#define ENCODED(CharT, S) chooseLiteral<CharT>(S, L##S)

TYPED_TEST(LlvmLibcStringViewTest, InitializeCheck, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView S;
  ASSERT_EQ(S.size(), size_t(0));
  ASSERT_TRUE(S.data() == nullptr);

  S = StringView(ENCODED(CharT, ""));
  ASSERT_EQ(S.size(), size_t(0));
  ASSERT_TRUE(S.data() != nullptr);

  S = StringView(ENCODED(CharT, "abc"), 0);
  ASSERT_EQ(S.size(), size_t(0));
  ASSERT_TRUE(S.data() != nullptr);

  S = StringView(ENCODED(CharT, "123456789"));
  ASSERT_EQ(S.size(), size_t(9));

  S = StringView(ENCODED(CharT, "123456789"), 5);
  ASSERT_EQ(S.size(), size_t(5));
}

TYPED_TEST(LlvmLibcStringViewTest, Equals, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView S(ENCODED(CharT, "abc"));
  ASSERT_EQ(S, StringView(ENCODED(CharT, "abc")));
  ASSERT_NE(S, StringView());
  ASSERT_NE(S, StringView(ENCODED(CharT, "")));
  ASSERT_NE(S, StringView(ENCODED(CharT, "123")));
  ASSERT_NE(S, StringView(ENCODED(CharT, "abd")));
  ASSERT_NE(S, StringView(ENCODED(CharT, "aaa")));
  ASSERT_NE(S, StringView(ENCODED(CharT, "abcde")));
}

TYPED_TEST(LlvmLibcStringViewTest, startsWith, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView S(ENCODED(CharT, "abc"));
  ASSERT_TRUE(S.starts_with(ENCODED(CharT, 'a')));
  ASSERT_TRUE(S.starts_with(StringView(ENCODED(CharT, "a"))));
  ASSERT_TRUE(S.starts_with(StringView(ENCODED(CharT, "ab"))));
  ASSERT_TRUE(S.starts_with(StringView(ENCODED(CharT, "abc"))));
  ASSERT_TRUE(S.starts_with(StringView()));
  ASSERT_TRUE(S.starts_with(StringView(ENCODED(CharT, ""))));
  ASSERT_FALSE(S.starts_with(ENCODED(CharT, '1')));
  ASSERT_FALSE(S.starts_with(StringView(ENCODED(CharT, "123"))));
  ASSERT_FALSE(S.starts_with(StringView(ENCODED(CharT, "abd"))));
  ASSERT_FALSE(S.starts_with(StringView(ENCODED(CharT, "aaa"))));
  ASSERT_FALSE(S.starts_with(StringView(ENCODED(CharT, "abcde"))));
}

TYPED_TEST(LlvmLibcStringViewTest, endsWith, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView S(ENCODED(CharT, "abc"));
  ASSERT_TRUE(S.ends_with(ENCODED(CharT, 'c')));
  ASSERT_TRUE(S.ends_with(StringView(ENCODED(CharT, "c"))));
  ASSERT_TRUE(S.ends_with(StringView(ENCODED(CharT, "bc"))));
  ASSERT_TRUE(S.ends_with(StringView(ENCODED(CharT, "abc"))));
  ASSERT_TRUE(S.ends_with(StringView()));
  ASSERT_TRUE(S.ends_with(StringView(ENCODED(CharT, ""))));
  ASSERT_FALSE(S.ends_with(ENCODED(CharT, '1')));
  ASSERT_FALSE(S.ends_with(StringView(ENCODED(CharT, "123"))));
  ASSERT_FALSE(S.ends_with(StringView(ENCODED(CharT, "abd"))));
  ASSERT_FALSE(S.ends_with(StringView(ENCODED(CharT, "aaa"))));
  ASSERT_FALSE(S.ends_with(StringView(ENCODED(CharT, "abcde"))));
}

TYPED_TEST(LlvmLibcStringViewTest, RemovePrefix, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView A(ENCODED(CharT, "123456789"));
  A.remove_prefix(0);
  ASSERT_EQ(A.size(), size_t(9));
  ASSERT_TRUE(A == ENCODED(CharT, "123456789"));

  StringView B(ENCODED(CharT, "123456789"));
  B.remove_prefix(4);
  ASSERT_EQ(B.size(), size_t(5));
  ASSERT_TRUE(B == ENCODED(CharT, "56789"));

  StringView C(ENCODED(CharT, "123456789"));
  C.remove_prefix(9);
  ASSERT_EQ(C.size(), size_t(0));
}

TYPED_TEST(LlvmLibcStringViewTest, RemoveSuffix, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView A(ENCODED(CharT, "123456789"));
  A.remove_suffix(0);
  ASSERT_EQ(A.size(), size_t(9));
  ASSERT_TRUE(A == ENCODED(CharT, "123456789"));

  StringView B(ENCODED(CharT, "123456789"));
  B.remove_suffix(4);
  ASSERT_EQ(B.size(), size_t(5));
  ASSERT_TRUE(B == ENCODED(CharT, "12345"));

  StringView C(ENCODED(CharT, "123456789"));
  C.remove_suffix(9);
  ASSERT_EQ(C.size(), size_t(0));
}

TYPED_TEST(LlvmLibcStringViewTest, Observer, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView ABC(ENCODED(CharT, "abc"));
  ASSERT_EQ(ABC.size(), size_t(3));
  ASSERT_FALSE(ABC.empty());
  ASSERT_EQ(ABC.front(), ENCODED(CharT, 'a'));
  ASSERT_EQ(ABC.back(), ENCODED(CharT, 'c'));
}

TYPED_TEST(LlvmLibcStringViewTest, FindFirstOf, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView Tmp(ENCODED(CharT, "abca"));
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'a')) == 0);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'd')) == StringView::npos);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'b')) == 1);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'a'), 0) == 0);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'b'), 1) == 1);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'a'), 1) == 3);
  ASSERT_TRUE(Tmp.find_first_of(ENCODED(CharT, 'a'), 42) == StringView::npos);
  ASSERT_FALSE(Tmp.find_first_of(ENCODED(CharT, 'c')) == 1);
  ASSERT_FALSE(Tmp.find_first_of(ENCODED(CharT, 'c'), 0) == 1);
  ASSERT_FALSE(Tmp.find_first_of(ENCODED(CharT, 'c'), 1) == 1);
}

TYPED_TEST(LlvmLibcStringViewTest, FindLastOf, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView Tmp(ENCODED(CharT, "abada"));

  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a')), size_t(4));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 123), size_t(4));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 5), size_t(4));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 4), size_t(4));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 3), size_t(2));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 2), size_t(2));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 1), size_t(0));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'a'), 0), size_t(0));

  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b')), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 123), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 5), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 4), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 3), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 2), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 1), size_t(1));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'b'), 0), StringView::npos);

  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd')), size_t(3));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 123), size_t(3));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 5), size_t(3));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 4), size_t(3));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 3), size_t(3));
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 2), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 1), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'd'), 0), StringView::npos);

  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e')), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 123), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 5), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 4), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 3), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 2), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 1), StringView::npos);
  ASSERT_EQ(Tmp.find_last_of(ENCODED(CharT, 'e'), 0), StringView::npos);

  StringView Empty;
  ASSERT_EQ(Empty.find_last_of(ENCODED(CharT, 'a')), StringView::npos);
  ASSERT_EQ(Empty.find_last_of(ENCODED(CharT, 'a'), 0), StringView::npos);
  ASSERT_EQ(Empty.find_last_of(ENCODED(CharT, 'a'), 123), StringView::npos);

  StringView Empty1(ENCODED(CharT, ""));
  ASSERT_EQ(Empty1.find_last_of(ENCODED(CharT, 'a')), StringView::npos);
  ASSERT_EQ(Empty1.find_last_of(ENCODED(CharT, 'a'), 0), StringView::npos);
  ASSERT_EQ(Empty1.find_last_of(ENCODED(CharT, 'a'), 123), StringView::npos);
}

TYPED_TEST(LlvmLibcStringViewTest, FindFirstNotOf, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView Tmp(ENCODED(CharT, "abada"));

  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a')), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 123), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 5), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 4), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 2), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'a'), 0), size_t(1));

  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b')), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 123), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 5), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 1), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'b'), 0), size_t(0));

  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd')), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 123), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 5), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 3), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'd'), 0), size_t(0));

  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e')), size_t(0));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 123), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 5), StringView::npos);
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 4), size_t(4));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 3), size_t(3));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 2), size_t(2));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 1), size_t(1));
  EXPECT_EQ(Tmp.find_first_not_of(ENCODED(CharT, 'e'), 0), size_t(0));

  StringView Empty;
  EXPECT_EQ(Empty.find_first_not_of(ENCODED(CharT, 'a')), StringView::npos);
  EXPECT_EQ(Empty.find_first_not_of(ENCODED(CharT, 'a'), 0), StringView::npos);
  EXPECT_EQ(Empty.find_first_not_of(ENCODED(CharT, 'a'), 123),
            StringView::npos);

  StringView Empty1(ENCODED(CharT, ""));
  EXPECT_EQ(Empty1.find_first_not_of(ENCODED(CharT, 'a')), StringView::npos);
  EXPECT_EQ(Empty1.find_first_not_of(ENCODED(CharT, 'a'), 0), StringView::npos);
  EXPECT_EQ(Empty1.find_first_not_of(ENCODED(CharT, 'a'), 123),
            StringView::npos);

  StringView Full(ENCODED(CharT, "aaaaaaa"));
  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'a')), StringView::npos);
  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'a'), 0), StringView::npos);
  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'a'), 123), StringView::npos);

  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'b')), size_t(0));
  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'b'), 0), size_t(0));
  EXPECT_EQ(Full.find_first_not_of(ENCODED(CharT, 'b'), 123), StringView::npos);
}

TYPED_TEST(LlvmLibcStringViewTest, Contains, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView Empty;
  EXPECT_FALSE(Empty.contains(ENCODED(CharT, 'a')));
  EXPECT_FALSE(Empty.contains(ENCODED(CharT, 'g')));
  EXPECT_FALSE(Empty.contains(ENCODED(CharT, 'q')));

  StringView S = ENCODED(CharT, "abada");
  EXPECT_TRUE(S.contains(ENCODED(CharT, 'a')));
  EXPECT_TRUE(S.contains(ENCODED(CharT, 'b')));
  EXPECT_FALSE(S.contains(ENCODED(CharT, 'c')));
  EXPECT_TRUE(S.contains(ENCODED(CharT, 'd')));
  EXPECT_FALSE(S.contains(ENCODED(CharT, 'e')));
}

TYPED_TEST(LlvmLibcStringViewTest, Substr, TestCharTypes) {
  using CharT = ParamType;
  using StringView = LIBC_NAMESPACE::cpp::basic_string_view<CharT>;

  StringView S = ENCODED(CharT, "abada");
  EXPECT_EQ(S.substr(0), StringView(ENCODED(CharT, "abada")));
  EXPECT_EQ(S.substr(1), StringView(ENCODED(CharT, "bada")));
  EXPECT_EQ(S.substr(3), StringView(ENCODED(CharT, "da")));
  EXPECT_EQ(S.substr(5), StringView(ENCODED(CharT, "")));
  EXPECT_EQ(S.substr(1, 3), StringView(ENCODED(CharT, "bad")));
  EXPECT_EQ(S.substr(3, 1), StringView(ENCODED(CharT, "d")));
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
