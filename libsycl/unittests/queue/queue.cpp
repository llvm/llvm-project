//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, CommonQueriesAndLifetime) {
  mock::MockWrapper Mock;

  EXPECT_CALL(Mock.get(), olCreateQueue(_, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyQueue(_)).Times(1);
  {
    queue Q;
    EXPECT_EQ(Q.get_backend(), sycl::backend::level_zero);
    EXPECT_EQ(Q.is_in_order(), false);
  }
}
