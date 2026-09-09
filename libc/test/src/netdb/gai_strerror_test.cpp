//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for gai_strerror.
///
//===----------------------------------------------------------------------===//

#include "hdr/netdb_macros.h"
#include "src/netdb/gai_strerror.h"
#include "test/UnitTest/Test.h"

TEST(LlvmLibcGaiStrerrorTest, AllKnownErrors) {
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_AGAIN),
               "Name could not be resolved at this time");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_BADFLAGS),
               "Flags had an invalid value");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_FAIL),
               "Non-recoverable error occurred");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_FAMILY),
               "Address family not recognized");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_MEMORY),
               "Memory allocation failure");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_NONAME),
               "Name does not resolve for the supplied parameters");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_OVERFLOW),
               "Argument buffer overflowed");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_SERVICE),
               "Service not recognized for specified socket type");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_SOCKTYPE),
               "Intended socket type not recognized");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(EAI_SYSTEM), "System error");
}

TEST(LlvmLibcGaiStrerrorTest, UnknownErrors) {
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(100), "Unknown error");
  EXPECT_STREQ(LIBC_NAMESPACE::gai_strerror(-1000), "Unknown error");
}
