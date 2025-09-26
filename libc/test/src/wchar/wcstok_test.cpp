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

TEST(LlvmLibcWCSTokReentrantTest, NoTokenFound) {
  { // Empty source and delimiter string.
    wchar_t empty[] = L"";
    wchar_t *reserve = nullptr;
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"", &reserve), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"", &reserve), nullptr);
    // Subsequent searches still return nullptr.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"", &reserve), nullptr);
  }
  { // Empty source and single character delimiter string.
    wchar_t empty[] = L"";
    wchar_t *reserve = nullptr;
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"_", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(empty, L"_", &reserve), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"_", &reserve), nullptr);
    // Subsequent searches still return nullptr.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"_", &reserve), nullptr);
  }
  { // Same character source and delimiter string.
    wchar_t single[] = L"_";
    wchar_t *reserve = nullptr;
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(single, L"_", &reserve), nullptr);
    // Another call to ensure that 'reserve' is not in a bad state.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(single, L"_", &reserve), nullptr);
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"_", &reserve), nullptr);
    // Subsequent searches still return nullptr.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L"_", &reserve), nullptr);
  }
  { // Multiple character source and single character delimiter string.
    wchar_t multiple[] = L"1,2";
    wchar_t *reserve = nullptr;
    wchar_t *tok = LIBC_NAMESPACE::wcstok(multiple, L":", &reserve);
    ASSERT_TRUE(tok[0] == L'1');
    ASSERT_TRUE(tok[1] == L',');
    ASSERT_TRUE(tok[2] == L'2');
    ASSERT_TRUE(tok[3] == L'\0');
    // Another call to ensure that 'reserve' is not in a bad state.
    tok = LIBC_NAMESPACE::wcstok(multiple, L":", &reserve);
    ASSERT_TRUE(tok[0] == L'1');
    ASSERT_TRUE(tok[1] == L',');
    ASSERT_TRUE(tok[2] == L'2');
    ASSERT_TRUE(tok[3] == L'\0');
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L":", &reserve), nullptr);
    // Subsequent searches still return nullptr.
    ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L":", &reserve), nullptr);
  }
}

TEST(LlvmLibcWCSTokReentrantTest, DelimiterAsFirstCharacterShouldBeIgnored) {
  wchar_t src[] = L".123";
  wchar_t *reserve = nullptr;
  wchar_t *tok = LIBC_NAMESPACE::wcstok(src, L".", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'3');
  ASSERT_TRUE(tok[3] == L'\0');
  // Another call to ensure that 'reserve' is not in a bad state.
  tok = LIBC_NAMESPACE::wcstok(src, L".", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'3');
  ASSERT_TRUE(tok[3] == L'\0');
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L".", &reserve), nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest, DelimiterIsMiddleCharacter) {
  wchar_t src[] = L"12,34";
  wchar_t *reserve = nullptr;
  wchar_t *tok = LIBC_NAMESPACE::wcstok(src, L",", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'\0');
  // Another call to ensure that 'reserve' is not in a bad state.
  tok = LIBC_NAMESPACE::wcstok(src, L",", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'\0');
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L",", &reserve), nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest, DelimiterAsLastCharacterShouldBeIgnored) {
  wchar_t src[] = L"1234:";
  wchar_t *reserve = nullptr;
  wchar_t *tok = LIBC_NAMESPACE::wcstok(src, L":", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'3');
  ASSERT_TRUE(tok[3] == L'4');
  ASSERT_TRUE(tok[4] == L'\0');
  // Another call to ensure that 'reserve' is not in a bad state.
  tok = LIBC_NAMESPACE::wcstok(src, L":", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'3');
  ASSERT_TRUE(tok[3] == L'4');
  ASSERT_TRUE(tok[4] == L'\0');
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L":", &reserve), nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest, ShouldNotGoPastNullTerminator) {
  wchar_t src[] = {L'1', L'2', L'\0', L',', L'3'};
  wchar_t *reserve = nullptr;
  wchar_t *tok = LIBC_NAMESPACE::wcstok(src, L",", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'\0');
  // Another call to ensure that 'reserve' is not in a bad state.
  tok = LIBC_NAMESPACE::wcstok(src, L",", &reserve);
  ASSERT_TRUE(tok[0] == L'1');
  ASSERT_TRUE(tok[1] == L'2');
  ASSERT_TRUE(tok[2] == L'\0');
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(nullptr, L",", &reserve), nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest,
     ShouldReturnNullptrWhenBothSrcAndSaveptrAreNull) {
  wchar_t *src = nullptr;
  wchar_t *reserve = nullptr;
  // Ensure that instead of crashing if src and reserve are null, nullptr is
  // returned
  ASSERT_EQ(LIBC_NAMESPACE::wcstok(src, L",", &reserve), nullptr);
  // And that neither src nor reserve are changed when that happens
  ASSERT_EQ(src, nullptr);
  ASSERT_EQ(reserve, nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest,
     SubsequentCallsShouldFindFollowingDelimiters) {
  wchar_t src[] = L"12,34.56";
  wchar_t *reserve = nullptr;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L",.", &reserve);
  ASSERT_TRUE(token[0] == L'1');
  ASSERT_TRUE(token[1] == L'2');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L",.", &reserve);
  ASSERT_TRUE(token[0] == L'3');
  ASSERT_TRUE(token[1] == L'4');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L",.", &reserve);
  ASSERT_TRUE(token[0] == L'5');
  ASSERT_TRUE(token[1] == L'6');
  ASSERT_TRUE(token[2] == L'\0');
  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &reserve);
  ASSERT_EQ(token, nullptr);
  // Subsequent calls after hitting the end of the string should also return
  // nullptr.
  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &reserve);
  ASSERT_EQ(token, nullptr);
}

TEST(LlvmLibcWCSTokReentrantTest, DelimitersShouldNotBeIncludedInToken) {
  wchar_t src[] = L"__ab__:_cd__:__ef__:__";
  wchar_t *reserve = nullptr;
  wchar_t *token = LIBC_NAMESPACE::wcstok(src, L"_:", &reserve);
  ASSERT_TRUE(token[0] == L'a');
  ASSERT_TRUE(token[1] == L'b');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L":_", &reserve);
  ASSERT_TRUE(token[0] == L'c');
  ASSERT_TRUE(token[1] == L'd');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,", &reserve);
  ASSERT_TRUE(token[0] == L'e');
  ASSERT_TRUE(token[1] == L'f');
  ASSERT_TRUE(token[2] == L'\0');

  token = LIBC_NAMESPACE::wcstok(nullptr, L"_:,_", &reserve);
  ASSERT_EQ(token, nullptr);
}
