//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unittests for capset.
///
//===----------------------------------------------------------------------===//

#include "hdr/errno_macros.h"
#include "hdr/types/cap_user_data_t.h"
#include "hdr/types/cap_user_header_t.h"
#include "src/sys/capability/capget.h"
#include "src/sys/capability/capset.h"
#include "test/UnitTest/ErrnoCheckingTest.h"
#include "test/UnitTest/ErrnoSetterMatcher.h"
#include "test/UnitTest/Test.h"

#include <linux/capability.h>

using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::any_of;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails;
using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
using LlvmLibcCapsetTest = LIBC_NAMESPACE::testing::ErrnoCheckingTest;

TEST_F(LlvmLibcCapsetTest, Current) {
  __user_cap_header_struct hdr = {_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
  ASSERT_THAT(LIBC_NAMESPACE::capget(&hdr, data), Succeeds(0));

  int ret = LIBC_NAMESPACE::capset(&hdr, data);
  if (ret == 0) {
    ASSERT_ERRNO_SUCCESS();
  } else {
    ASSERT_THAT(ret, Fails(any_of(EPERM, ENOSYS)));
  }
}

TEST_F(LlvmLibcCapsetTest, Failure) {
  __user_cap_header_struct bad_hdr = {0, 0};
  __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
  ASSERT_THAT(LIBC_NAMESPACE::capset(&bad_hdr, data),
              Fails(any_of(EINVAL, EPERM, ENOSYS)));

  __user_cap_header_struct hdr = {_LINUX_CAPABILITY_VERSION_3, 0};
  ASSERT_THAT(LIBC_NAMESPACE::capget(&hdr, data), Succeeds(0));
  if ((data[0].permitted & (1U << CAP_SETUID)) == 0) {
    data[0].effective |= (1U << CAP_SETUID);
    data[0].permitted |= (1U << CAP_SETUID);
    ASSERT_THAT(LIBC_NAMESPACE::capset(&hdr, data),
                Fails(any_of(EPERM, ENOSYS)));
  }
}
