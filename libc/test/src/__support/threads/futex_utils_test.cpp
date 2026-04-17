//===-- Unittests for futex utilities ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/optional.h"
#include "src/__support/threads/futex_utils.h"
#include "test/UnitTest/Test.h"

#include "hdr/errno_macros.h"

TEST(LlvmLibcSupportThreadsFutexUtilsTest, RequeueSmokeTest) {
  LIBC_NAMESPACE::Futex source(1);
  LIBC_NAMESPACE::Futex destination(2);

  auto no_requeue = source.requeue_to(destination, 1, 1, 0);
  if (no_requeue.has_value())
    ASSERT_EQ(*no_requeue, 0);
  else
    ASSERT_EQ(no_requeue.error(), ENOSYS);

  auto no_wake = source.requeue_to(destination, 1, 0, 1);
  if (no_wake.has_value())
    ASSERT_EQ(*no_wake, 0);
  else
    ASSERT_EQ(no_wake.error(), ENOSYS);

  auto cmp_mismatch = source.requeue_to(destination, 0, 0, 1);
  if (cmp_mismatch.has_value())
    ASSERT_EQ(*cmp_mismatch, 0);
  else
    ASSERT_TRUE(cmp_mismatch.error() == ENOSYS ||
                cmp_mismatch.error() == EAGAIN);
}
