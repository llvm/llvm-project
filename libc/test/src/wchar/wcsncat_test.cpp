//===-- Unittests for wcscat ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/wchar/wcsncat.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcWCSNCatTest, EmptyDest) {
  wchar_t dest[4] = {L'\0'};
  const wchar_t *src = L"abc";

  // Start by copying nothing
  LIBC_NAMESPACE::wcsncat(dest, src, 0);
  ASSERT_TRUE(dest[0] == L'\0');

  // Copying part of it.
  LIBC_NAMESPACE::wcsncat(dest, src, 1);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'\0');

  // Resetting for the last test.
  dest[0] = '\0';

  // Copying all of it.
  LIBC_NAMESPACE::wcsncat(dest, src, 3);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_TRUE(dest[2] == L'c');
  ASSERT_TRUE(dest[3] == L'\0');
}

TEST(LlvmLibcWCSNCatTest, NonEmptyDest) {
  wchar_t dest[7] = {L'x', L'y', L'z', L'\0'};
  const wchar_t *src = L"abc";

  // Adding on only part of the string
  LIBC_NAMESPACE::wcsncat(dest, src, 1);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_TRUE(dest[2] == L'z');
  ASSERT_TRUE(dest[3] == L'a');
  ASSERT_TRUE(dest[4] == L'\0');

  // Copying more without resetting
  LIBC_NAMESPACE::wcsncat(dest, src, 2);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_TRUE(dest[2] == L'z');
  ASSERT_TRUE(dest[3] == L'a');
  ASSERT_TRUE(dest[4] == L'a');
  ASSERT_TRUE(dest[5] == L'b');
  ASSERT_TRUE(dest[6] == L'\0');

  // Setting end marker to make sure it overwrites properly.
  dest[3] = L'\0';

  // Copying all of it.
  LIBC_NAMESPACE::wcsncat(dest, src, 3);
  ASSERT_TRUE(dest[0] == L'x');
  ASSERT_TRUE(dest[1] == L'y');
  ASSERT_TRUE(dest[2] == L'z');
  ASSERT_TRUE(dest[3] == L'a');
  ASSERT_TRUE(dest[4] == L'b');
  ASSERT_TRUE(dest[5] == L'c');
  ASSERT_TRUE(dest[6] == L'\0');

  // Check that copying still works when count > src length.
  dest[0] = L'\0';
  // And that it doesn't write beyond what is necessary.
  dest[4] = L'Z';
  LIBC_NAMESPACE::wcsncat(dest, src, 4);
  ASSERT_TRUE(dest[0] == L'a');
  ASSERT_TRUE(dest[1] == L'b');
  ASSERT_TRUE(dest[2] == L'c');
  ASSERT_TRUE(dest[3] == L'\0');
  ASSERT_TRUE(dest[4] == L'Z');
}
