//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/helpers.hpp>

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

using namespace sycl;

// inline ur_result_t redefinedEventsWaitWithBarrier(void *pParams) {
//   GEventsWaitCounter++;
//   return UR_RESULT_SUCCESS;
// }

TEST(PlatformTest, APIGetPlatformsDefaultMock) {
  // Use default callbacks.
  unittest::OffloadMock mock;
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1);
  EXPECT_EQ(Platforms[0].get_backend(), sycl::backend::level_zero);

  auto Devices = Platforms[0].get_devices();
  ASSERT_EQ(Devices.size(), 1);
  EXPECT_EQ(Devices[0].get_backend(), sycl::backend::level_zero);

  EXPECT_FALSE(Devices[0].is_cpu());
  EXPECT_FALSE(Devices[0].is_accelerator());
  EXPECT_TRUE(Devices[0].is_gpu());

  EXPECT_EQ(Devices[0].get_platform(), Platforms[0]);

  // mock::getCallbacks().set_before_callback("urEnqueueEventsWaitWithBarrier",
  //                                          &redefinedEventsWaitWithBarrier);

  // context Ctx{Plt};
  // queue InOrderQueue{Ctx, default_selector_v, property::queue::in_order()};

  // auto buf = sycl::malloc_device<int>(1, InOrderQueue);
  // event Evt = InOrderQueue.submit(
  //     [&](sycl::handler &CGH) { CGH.memset(buf, 0, sizeof(buf[0])); });
  // InOrderQueue.submit([&](sycl::handler &CGH) { CGH.host_task([=] {}); })
  //     .wait();

  // size_t expectedCount = 1u;
  // EXPECT_EQ(GEventsWaitCounter, expectedCount);
}
