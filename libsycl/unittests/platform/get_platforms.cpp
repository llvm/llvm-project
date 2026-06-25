//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

using namespace sycl;

TEST(PlatformTest, APIGetPlatformsDefaultMock) {
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);
  EXPECT_EQ(Platforms[0].get_backend(), sycl::backend::level_zero);

  auto Devices = Platforms[0].get_devices();
  ASSERT_EQ(Devices.size(), 1u);
  EXPECT_EQ(Devices[0].get_backend(), sycl::backend::level_zero);

  EXPECT_FALSE(Devices[0].is_cpu());
  EXPECT_FALSE(Devices[0].is_accelerator());
  EXPECT_TRUE(Devices[0].is_gpu());

  EXPECT_EQ(Devices[0].get_platform(), Platforms[0]);
}
