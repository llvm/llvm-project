//===-- Unittests for struct sockaddr_storage -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hdr/types/struct_sockaddr_storage.h"
#include "hdr/types/struct_sockaddr_un.h"

#include "test/UnitTest/LibcTest.h"

#include <sys/socket.h> // For AF_UNIX

using LlvmLibcSockaddrStorageTest = LIBC_NAMESPACE::testing::Test;

sa_family_t test_sockaddr_aliasing(struct sockaddr_storage *ss,
                                   struct sockaddr_un *sun);

TEST_F(LlvmLibcSockaddrStorageTest, SizeAndAlignment) {
  // TODO: Add other sockaddr_* types as they are defined.
  static_assert(sizeof(struct sockaddr_un) <= sizeof(struct sockaddr_storage));
  static_assert(alignof(struct sockaddr_un) <=
                alignof(struct sockaddr_storage));
}

// Test only makes sense in the full build, as otherwise we're testing the
// system type definitions (and some of those don't handle aliasing properly).
#if defined(LIBC_FULL_BUILD)
TEST_F(LlvmLibcSockaddrStorageTest, MemberAccess) {
  struct sockaddr_storage ss = {};
  auto *sun = reinterpret_cast<struct sockaddr_un *>(&ss);
  ASSERT_EQ(static_cast<sa_family_t>(AF_UNIX),
            test_sockaddr_aliasing(&ss, sun));
}
#endif
