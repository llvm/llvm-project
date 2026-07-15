//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/scoped_binary_registration.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>

using namespace sycl;
using namespace ::testing;

class sycl::detail::MockQueue : public sycl::queue {
public:
  using sycl::queue::sycl_kernel_launch;
};

struct KernelData {
  char Arr[16];
  int SomeInt;
  double DoubleArr[8];
};

TEST(Queue, KernelLaunch) {
  mock::MockWrapper Mock;
  sycl::unittests::ScopedKernelRegistration Registration{"TestKernel"};
  sycl::detail::MockQueue Q;
  KernelData Data{};
  Data.Arr[0] = 'A';
  Data.Arr[1] = 'I';
  Data.SomeInt = 42;
  Data.DoubleArr[0] = 3.14;
  Data.DoubleArr[7] = -1.25;

  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _))
      .WillOnce([&Data](ol_queue_handle_t Queue, ol_device_handle_t Device,
                        ol_symbol_handle_t Kernel,
                        const ol_kernel_launch_size_args_t *LaunchSizeArgs,
                        const ol_kernel_launch_prop_t *Properties,
                        size_t NumArgs, void **ArgPtrs,
                        const size_t *ArgSizes) -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Kernel, nullptr);
        EXPECT_NE(LaunchSizeArgs, nullptr);
        std::ignore = Properties;
        EXPECT_EQ(NumArgs, 1u);
        EXPECT_NE(ArgPtrs, nullptr);
        EXPECT_NE(ArgSizes, nullptr);
        EXPECT_EQ(ArgSizes[0], sizeof(KernelData));

        auto PayloadPtr = static_cast<const KernelData *>(ArgPtrs[0]);
        EXPECT_NE(PayloadPtr, nullptr);
        EXPECT_EQ(std::memcmp(PayloadPtr, &Data, sizeof(KernelData)), 0);
        return OL_SUCCESS;
      });
  Q.sycl_kernel_launch<class TestKernel>("TestKernel", Data);

  EXPECT_CALL(Mock.get(), olSyncQueue(_)).Times(1);
  Q.wait();
}
