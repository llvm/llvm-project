//===-- Unittests for inet_aton -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/arpa/inet/htonl.h"
#include "src/arpa/inet/inet_aton.h"
#include "test/UnitTest/Test.h"

namespace LIBC_NAMESPACE_DECL {

TEST(LlvmLibcInetAton, ValidTest) {
  in_addr a;

  // a.b.c.d
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("127.1.2.4", &a));
  ASSERT_EQ(htonl(0x7f010204), a.s_addr);

  // a.b.c
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("127.1.4", &a));
  ASSERT_EQ(htonl(0x7f010004), a.s_addr);

  // a.b
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("127.1", &a));
  ASSERT_EQ(htonl(0x7f000001), a.s_addr);

  // a
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("0x7f000001", &a));
  ASSERT_EQ(htonl(0x7f000001), a.s_addr);

  // Hex (0x) and mixed-case hex digits.
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("0xFf.0.0.1", &a));
  ASSERT_EQ(htonl(0xff000001), a.s_addr);

  // Hex (0X) and mixed-case hex digits.
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("0XfF.0.0.1", &a));
  ASSERT_EQ(htonl(0xff000001), a.s_addr);

  // Octal.
  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("0177.0.0.1", &a));
  ASSERT_EQ(htonl(0x7f000001), a.s_addr);

  a.s_addr = 0;
  ASSERT_EQ(1, inet_aton("036", &a));
  ASSERT_EQ(htonl(036U), a.s_addr);
}

TEST(LlvmLibcInetAton, InvalidTest) {
  ASSERT_EQ(0, inet_aton("", nullptr));           // Empty.
  ASSERT_EQ(0, inet_aton("x", nullptr));          // Leading junk.
  ASSERT_EQ(0, inet_aton("127.0.0.1x", nullptr)); // Trailing junk.
  ASSERT_EQ(0, inet_aton("09.0.0.1", nullptr));   // Invalid octal.
  ASSERT_EQ(0, inet_aton("0xg.0.0.1", nullptr));  // Invalid hex.
  ASSERT_EQ(0, inet_aton("1.2.3.4.5", nullptr));  // Too many dots.
  ASSERT_EQ(0, inet_aton("1.2.3.4.", nullptr));   // Trailing dot.

  // Out of range a.b.c.d form.
  ASSERT_EQ(0, inet_aton("999.0.0.1", nullptr));
  ASSERT_EQ(0, inet_aton("0.999.0.1", nullptr));
  ASSERT_EQ(0, inet_aton("0.0.999.1", nullptr));
  ASSERT_EQ(0, inet_aton("0.0.0.999", nullptr));

  // Out of range a.b.c form.
  ASSERT_EQ(0, inet_aton("256.0.0", nullptr));
  ASSERT_EQ(0, inet_aton("0.256.0", nullptr));
  ASSERT_EQ(0, inet_aton("0.0.0x10000", nullptr));

  // Out of range a.b form.
  ASSERT_EQ(0, inet_aton("256.0", nullptr));
  ASSERT_EQ(0, inet_aton("0.0x1000000", nullptr));

  // Out of range a form.
  ASSERT_EQ(0, inet_aton("0x100000000", nullptr));

  // 64-bit overflow.
  ASSERT_EQ(0, inet_aton("0x10000000000000000", nullptr));

  // Out of range octal.
  ASSERT_EQ(0, inet_aton("0400.0.0.1", nullptr));
}

} // namespace LIBC_NAMESPACE_DECL
