//===-- Unittests for mbsinit ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/wchar_t.h"
#include "src/string/memset.h"
#include "src/wchar/mbrtowc.h"
#include "src/wchar/mbsinit.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcMBSInitTest, EmptyState) {
  mbstate_t ps;
  LIBC_NAMESPACE::memset(&ps, 0, sizeof(mbstate_t));
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&ps), 0);
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(nullptr), 0);
}

TEST(LlvmLibcMBSInitTest, ConversionTest) {
  const char *src = "\xf0\x9f\xa4\xa3"; // 4 byte emoji
  wchar_t dest[2];
  mbstate_t ps;
  LIBC_NAMESPACE::memset(&ps, 0, sizeof(mbstate_t));

  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&ps), 0);
  LIBC_NAMESPACE::mbrtowc(dest, src, 2, &ps); // partial conversion
  ASSERT_EQ(LIBC_NAMESPACE::mbsinit(&ps), 0);
  LIBC_NAMESPACE::mbrtowc(dest, src + 2, 2, &ps); // complete conversion
  ASSERT_NE(LIBC_NAMESPACE::mbsinit(&ps), 0);     // state should be reset now
}
