//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/device_images.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

using namespace sycl;
using namespace ::testing;

namespace {

class ScopedKernelRegistration {
public:
  explicit ScopedKernelRegistration(std::string KernelName)
      : MKernelName(std::move(KernelName)),
        MEntry{sycl::unittest::GenericEntry},
        MImage{sycl::unittest::GenericDeviceImage}, MBinary{},
        MDesc{sycl::unittest::GenericDeviceImages} {
    MEntry.SymbolName = MKernelName.data();
    MEntry.Size = MKernelName.size();

    MImage.EntriesBegin = &MEntry;
    MImage.EntriesEnd = &MEntry + 1;
    MImage.ImageStart = MBinary.data();
    MImage.ImageEnd = MBinary.data() + MBinary.size();

    MDesc.NumDeviceBinaries = 1;
    MDesc.DeviceImages = &MImage;

    sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(&MDesc);
  }

  ~ScopedKernelRegistration() {
    sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
        &MDesc);
  }

private:
  std::string MKernelName;
  llvm::offloading::EntryTy MEntry;
  sycl::detail::__sycl_tgt_device_image MImage;
  std::array<unsigned char, 4> MBinary;
  sycl::detail::__sycl_tgt_bin_desc MDesc;
};

} // namespace

TEST(Queue, CommonQueriesAndLifetime) {
  MockWrapper Mock;

  EXPECT_CALL(Mock.get(), olCreateQueue(_, _)).Times(1);
  EXPECT_CALL(Mock.get(), olDestroyQueue(_)).Times(1);
  {
    queue Q;
    EXPECT_EQ(Q.get_backend(), sycl::backend::level_zero);
    EXPECT_EQ(Q.is_in_order(), false);
  }
}

class sycl::detail::MockQueue : public sycl::queue {
  public:
  using sycl::queue::sycl_kernel_launch;
};

struct KernelData
{
  char Arr[16];
  int SomeInt;
  double DoubleArr[8];
};

TEST(Queue, SingleTask) {
  MockWrapper Mock;
  ScopedKernelRegistration Registration{"TestKernel"};
  sycl::detail::MockQueue Q;
  KernelData Data{};
  Data.Arr[0] = 'A';
  Data.Arr[1] = 'I';
  Data.SomeInt = 42;
  Data.DoubleArr[0] = 3.14;
  Data.DoubleArr[7] = -1.25;

  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, sizeof(KernelData), _))
      .WillOnce([&Data](ol_queue_handle_t Queue, ol_device_handle_t Device,
                        ol_symbol_handle_t Kernel, const void *ArgumentsData,
                        size_t ArgumentsSize,
                        const ol_kernel_launch_size_args_t *LaunchSizeArgs)
                    -> ol_result_t {
        EXPECT_NE(Queue, nullptr);
        EXPECT_NE(Device, nullptr);
        EXPECT_NE(Kernel, nullptr);
        EXPECT_NE(ArgumentsData, nullptr);
        EXPECT_EQ(ArgumentsSize, sizeof(KernelData));
        EXPECT_NE(LaunchSizeArgs, nullptr);

        auto PayloadPtr = static_cast<const KernelData *>(
            *static_cast<const void *const *>(ArgumentsData));
        EXPECT_NE(PayloadPtr, nullptr);
        EXPECT_EQ(std::memcmp(PayloadPtr, &Data, sizeof(KernelData)), 0);
        return OL_SUCCESS;
      });

  Q.sycl_kernel_launch<class TestKernel>("TestKernel", Data);
  Q.wait();
}
