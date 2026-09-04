//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getaddrinfo and freeaddrinfo.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/netdb_macros.h"
#include "hdr/types/struct_addrinfo.h"
#include "src/netdb/freeaddrinfo.h"
#include "src/netdb/getaddrinfo.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcNetdbTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcNetdbTest, GetAddrInfoReturnsEaiSystem) {
  struct addrinfo *res = nullptr;
  ASSERT_THAT(LIBC_NAMESPACE::getaddrinfo("localhost", nullptr, nullptr, &res),
              Fails(ENOSYS, EAI_SYSTEM));
  EXPECT_EQ(res, nullptr);

  // freeaddrinfo should do nothing and not crash.
  LIBC_NAMESPACE::freeaddrinfo(res);
}
