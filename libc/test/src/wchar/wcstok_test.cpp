//===-- Unittests for wcstok ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/wchar/wcstok.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcStrTokTest, NoTokenFound) {
  wchar_t empty[] = L"";
  wchar_t *buf;
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"", &buf), nullptr);
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"_", &buf), nullptr);

  wchar_t single[] = L"_";
  wchar_t *token = LIBC_NAMESPACE::wcstok(single, L"", &buf);
  ASSERT_TRUE(token[0] == L'_');
  ASSERT_TRUE(token[1] == L'\0');

  wchar_t multiple[] = L"1,2";
  token = LIBC_NAMESPACE::wcstok(multiple, L":", &buf);
  ASSERT_TRUE(multiple[0] == L'1');
  ASSERT_TRUE(multiple[1] == L',');
  ASSERT_TRUE(multiple[2] == L'2');
  ASSERT_TRUE(multiple[3] == L'\0');
}

TEST(LlvmLibcStrTokTest, DelimiterAsFirstCharacterShouldBeIgnored) {
  wchar_t *buf;
  wchar_t src[] = L".123";
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L".", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'3');
  ASSERT_TRUE(token[3] == L'\0');
}

TEST(LlvmLibcStrTokTest, DelimiterIsMiddleCharacter) {
  wchar_t src[] = L"12,34";
  wchar_t *buf;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L",", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');
}

TEST(LlvmLibcStrTokTest, DelimiterAsLastCharacterShouldBeIgnored) {
  wchar_t src[] = L"1234:";
  wchar_t *buf;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L":", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'3');
  ASSERT_TRUE(token[3] == L'4');
  ASSERT_TRUE(token[4] == L'\0');
}

TEST(LlvmLibcStrTokTest, MultipleDelimiters) {
  wchar_t src[] = L"12,.34";
  wchar_t *buf;
  wchar_t *token;

  token = LIBC_NAMESPACE::wcstok(src, L".", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L',');
  ASSERT_TRUE(token[3] == L'\0');

  token = LIBC_NAMESPACE::wcstok(src, L".,", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(src, L",.", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(src, L":,.", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');
}

TEST(LlvmLibcStrTokTest, ShouldNotGoPastNullTerminator) {
  wchar_t src[] = {L'1', L'2', L'\0', L',', L'3'};
  wchar_t *buf;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L",", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');
}

TEST(LlvmLibcStrTokTest, SubsequentCallsShouldFindFollowingDelimiters) {
  wchar_t src[] = L"12,34.56";
  wchar_t *buf;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L",.", &buf);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L",.", &buf);
  ASSERT_TRUE(token[0] == L'3');
  ASSERT_TRUE(token[1] == L'4');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L",.", &buf);
  ASSERT_TRUE(token[0] == L'5');
  ASSERT_TRUE(token[1] == L'6');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &buf);
  ASSERT_EQ(token, nullptr);
  // Subsequent calls after hitting the end of the string should also return
  // nullptr.
  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &buf);
  ASSERT_EQ(token, nullptr);
}

TEST(LlvmLibcStrTokTest, DelimitersShouldNotBeIncludedInToken) {
  wchar_t *buf;
  wchar_t src[] = L"__ab__:_cd__:__ef__:__";
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L"_:", &buf);
  ASSERT_TRUE(token[0] == L'a');
  ASSERT_TRUE(token[1] == L'b');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L":_", &buf);
  ASSERT_TRUE(token[0] == L'c');
  ASSERT_TRUE(token[1] == L'd');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,", &buf);
  ASSERT_TRUE(token[0] == L'e');
  ASSERT_TRUE(token[1] == L'f');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &buf);
  ASSERT_EQ(token, nullptr);
}
