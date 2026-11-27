//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for bounded cpp::span helpers in string_utils.h.
///
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/string/string_utils.h"
#include "test/UnitTest/Test.h"

static constexpr bool check_constexpr_tokens() {
  char ws_buf[] = "  alpha\t\tbeta ";
  LIBC_NAMESPACE::cpp::span<char> ws_rest(ws_buf);
  auto a = LIBC_NAMESPACE::internal::string_token(ws_rest, " \t");
  auto b = LIBC_NAMESPACE::internal::string_token(ws_rest, " \t");
  auto c = LIBC_NAMESPACE::internal::string_token(ws_rest, " \t");
  if (!a || !b || c ||
      LIBC_NAMESPACE::cpp::string_view(a->data(), a->size()) != "alpha" ||
      LIBC_NAMESPACE::cpp::string_view(b->data(), b->size()) != "beta")
    return false;

  char sep_buf[] = ":x:";
  LIBC_NAMESPACE::cpp::span<char> sep_rest(sep_buf);
  auto f1 = LIBC_NAMESPACE::internal::string_token<false>(sep_rest, ":");
  auto f2 = LIBC_NAMESPACE::internal::string_token<false>(sep_rest, ":");
  auto f3 = LIBC_NAMESPACE::internal::string_token<false>(sep_rest, ":");
  auto f4 = LIBC_NAMESPACE::internal::string_token<false>(sep_rest, ":");
  return f1 && f1->empty() && f2 &&
         LIBC_NAMESPACE::cpp::string_view(f2->data(), f2->size()) == "x" &&
         f3 && f3->empty() && !f4;
}

static_assert(check_constexpr_tokens());

TEST(LlvmLibcStringUtilsTest, StringTokenSkipDelim) {
  char buf[] = "  tcp\t\t6  TCP\r";
  LIBC_NAMESPACE::cpp::span<char> rest(buf);
  constexpr LIBC_NAMESPACE::cpp::string_view WS = " \t\r";

  auto t1 = LIBC_NAMESPACE::internal::string_token(rest, WS);
  ASSERT_TRUE(t1.has_value());
  EXPECT_EQ(t1->size(), size_t(3));
  EXPECT_STREQ(t1->data(), "tcp");

  auto t2 = LIBC_NAMESPACE::internal::string_token(rest, WS);
  ASSERT_TRUE(t2.has_value());
  EXPECT_EQ(t2->size(), size_t(1));
  EXPECT_STREQ(t2->data(), "6");

  auto t3 = LIBC_NAMESPACE::internal::string_token(rest, WS);
  ASSERT_TRUE(t3.has_value());
  EXPECT_EQ(t3->size(), size_t(3));
  EXPECT_STREQ(t3->data(), "TCP");

  EXPECT_FALSE(LIBC_NAMESPACE::internal::string_token(rest, WS).has_value());
  EXPECT_TRUE(rest.empty());

  // Empty string and empty delimiter set.
  char empty_str[] = "";
  LIBC_NAMESPACE::cpp::span<char> empty_rest(empty_str);
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token(empty_rest, WS).has_value());
  EXPECT_TRUE(empty_rest.empty());

  char whole[] = "abc def";
  LIBC_NAMESPACE::cpp::span<char> whole_rest(whole);
  auto single = LIBC_NAMESPACE::internal::string_token(whole_rest, "");
  ASSERT_TRUE(single.has_value());
  EXPECT_EQ(single->size(), size_t(7));
  EXPECT_STREQ(single->data(), "abc def");
  EXPECT_TRUE(whole_rest.empty());

  // Delimiter set containing NUL must not skip past NUL terminators.
  char nul_delim_buf[] = " \0after";
  LIBC_NAMESPACE::cpp::span<char> nul_delim_rest(nul_delim_buf);
  constexpr LIBC_NAMESPACE::cpp::string_view NUL_DELIM("\0 ", 2);
  EXPECT_FALSE(LIBC_NAMESPACE::internal::string_token(nul_delim_rest, NUL_DELIM)
                   .has_value());
  EXPECT_TRUE(nul_delim_rest.empty());
}

TEST(LlvmLibcStringUtilsTest, StringTokenPreserveEmpty) {
  char buf[] = ":a::c:";
  LIBC_NAMESPACE::cpp::span<char> rest(buf);

  const char *expected[] = {"", "a", "", "c", ""};
  for (const char *exp : expected) {
    auto field = LIBC_NAMESPACE::internal::string_token<false>(rest, ":");
    ASSERT_TRUE(field.has_value());
    EXPECT_STREQ(field->data(), exp);
    EXPECT_EQ(field->size(), LIBC_NAMESPACE::internal::string_length(exp));
  }
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token<false>(rest, ":").has_value());
  EXPECT_TRUE(rest.empty());

  // Single empty string yields one empty field, matching strsep.
  char empty_str[] = "";
  LIBC_NAMESPACE::cpp::span<char> empty_rest(empty_str);
  auto empty_field =
      LIBC_NAMESPACE::internal::string_token<false>(empty_rest, ":");
  ASSERT_TRUE(empty_field.has_value());
  EXPECT_EQ(empty_field->size(), size_t(0));
  EXPECT_STREQ(empty_field->data(), "");
  EXPECT_TRUE(empty_rest.empty());

  // Embedded NUL terminates early without crossing into trailing bytes.
  char emb[] = "a\0:b";
  LIBC_NAMESPACE::cpp::span<char> emb_rest(emb);
  auto first = LIBC_NAMESPACE::internal::string_token<false>(emb_rest, ":");
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->size(), size_t(1));
  EXPECT_STREQ(first->data(), "a");
  EXPECT_TRUE(emb_rest.empty());
}

TEST(LlvmLibcStringUtilsTest, StringTokenBoundsAndNulSafety) {
  // Empty and unterminated spans return nullopt without modifying `rest`.
  LIBC_NAMESPACE::cpp::span<char> empty_span;
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token(empty_span, " ").has_value());
  EXPECT_FALSE(LIBC_NAMESPACE::internal::string_token<false>(empty_span, ":")
                   .has_value());

  char unterminated[3] = {'a', ' ', 'b'};
  LIBC_NAMESPACE::cpp::span<char> bad(unterminated);
  EXPECT_FALSE(LIBC_NAMESPACE::internal::string_token(bad, " ").has_value());
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token<false>(bad, " ").has_value());
  EXPECT_EQ(bad.size(), size_t(3));
  EXPECT_EQ(unterminated[1], ' ');

  // Delimiter-only input returns nullopt when SkipDelim is true.
  char spaces[] = "   \t ";
  LIBC_NAMESPACE::cpp::span<char> ws_rest(spaces);
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token(ws_rest, " \t").has_value());
  EXPECT_TRUE(ws_rest.empty());

  // Embedded NUL terminates tokenization early without reading past it.
  char embedded[] = "  \0hidden";
  LIBC_NAMESPACE::cpp::span<char> emb_rest(embedded);
  EXPECT_FALSE(
      LIBC_NAMESPACE::internal::string_token(emb_rest, " ").has_value());
  EXPECT_TRUE(emb_rest.empty());

  char mid_nul[] = "ok\0tail";
  LIBC_NAMESPACE::cpp::span<char> mid_rest(mid_nul);
  auto tok = LIBC_NAMESPACE::internal::string_token(mid_rest, " ");
  ASSERT_TRUE(tok.has_value());
  EXPECT_EQ(tok->size(), size_t(2));
  EXPECT_STREQ(tok->data(), "ok");
  EXPECT_TRUE(mid_rest.empty());
}

TEST(LlvmLibcStringUtilsTest, SpanStrlcpyStrnlenAndFindFirstChar) {
  char dst[5];
  EXPECT_EQ(LIBC_NAMESPACE::internal::strlcpy(dst, "abcdef"), size_t(6));
  EXPECT_STREQ(dst, "abcd");
  EXPECT_EQ(LIBC_NAMESPACE::internal::strnlen(dst), size_t(4));

  // Exact fit (len == dst.size() - 1) and 1-byte destination.
  EXPECT_EQ(LIBC_NAMESPACE::internal::strlcpy(dst, "wxyz"), size_t(4));
  EXPECT_STREQ(dst, "wxyz");

  char one[1] = {'?'};
  EXPECT_EQ(LIBC_NAMESPACE::internal::strlcpy(one, "hi"), size_t(2));
  EXPECT_EQ(one[0], '\0');

  LIBC_NAMESPACE::cpp::span<char> empty_dst;
  EXPECT_EQ(LIBC_NAMESPACE::internal::strlcpy(empty_dst, "hi"), size_t(2));

  // strnlen on unterminated, empty, and embedded-NUL spans.
  const char unterminated[3] = {'x', 'y', 'z'};
  EXPECT_EQ(LIBC_NAMESPACE::internal::strnlen(unterminated), size_t(3));
  EXPECT_EQ(LIBC_NAMESPACE::internal::strnlen(
                LIBC_NAMESPACE::cpp::span<const char>()),
            size_t(0));
  const char emb_nul[] = "ab\0cd";
  EXPECT_EQ(LIBC_NAMESPACE::internal::strnlen(emb_nul), size_t(2));

  // find_first_character match, miss, and empty span.
  unsigned char bytes[] = {'a', 'b', 'c', 'b'};
  EXPECT_EQ(LIBC_NAMESPACE::internal::find_first_character(bytes, 'b'),
            static_cast<void *>(bytes + 1));
  EXPECT_EQ(LIBC_NAMESPACE::internal::find_first_character(bytes, 'z'),
            nullptr);
  EXPECT_EQ(LIBC_NAMESPACE::internal::find_first_character({}, 'a'), nullptr);
}
