//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for getifaddrs and freeifaddrs.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/struct_ifaddrs.h"
#include "hdr/types/struct_sockaddr.h"
#include "src/ifaddrs/freeifaddrs.h"
#include "src/ifaddrs/getifaddrs.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

using namespace LIBC_NAMESPACE::testing::ErrnoSetterMatcher;
using LlvmLibcGetifaddrsTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcGetifaddrsTest, GetIfAddrsReturnsEnosys) {
  struct ifaddrs *ifap = nullptr;
  ASSERT_THAT(LIBC_NAMESPACE::getifaddrs(&ifap), Fails(ENOSYS));
  EXPECT_EQ(ifap, nullptr);

  // freeifaddrs should do nothing and not crash.
  LIBC_NAMESPACE::freeifaddrs(ifap);
}
