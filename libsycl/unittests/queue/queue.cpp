//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

TEST(Queue, CommonQueriesAndLifetime) {
  mock::MockWrapper Mock;

  EXPECT_CALL(Mock.get(), olCreateQueue(_, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyQueue(_)).Times(1);
  {
    queue Q;
    EXPECT_EQ(Q.get_backend(), sycl::backend::level_zero);
    EXPECT_EQ(Q.is_in_order(), false);
  }
}

TEST(Queue, ContextAndDeviceConstructor) {
  mock::MockWrapper Mock;

  const device Device;
  const context Context = Device.get_platform().khr_get_default_context();
  const auto Selector = [](const device &) { return 1; };
  const async_handler AsyncHandler = [](exception_list) {};
  EXPECT_CALL(Mock.get(), olCreateQueue(_, _, _)).Times(3);
  EXPECT_CALL(Mock.get(), olDestroyQueue(_)).Times(3);

  queue Queue(Context, Device);
  EXPECT_EQ(Queue.get_context(), Context);
  EXPECT_EQ(Queue.get_device(), Device);

  queue SelectorQueue(Context, Selector);
  EXPECT_EQ(SelectorQueue.get_context(), Context);
  EXPECT_EQ(SelectorQueue.get_device(), Device);

  queue AsyncSelectorQueue(Context, Selector, AsyncHandler);
  EXPECT_EQ(AsyncSelectorQueue.get_context(), Context);
  EXPECT_EQ(AsyncSelectorQueue.get_device(), Device);
}
