//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>
#include <common/unittests_helper.hpp>

#include <detail/device_impl.hpp>
#include <detail/program_manager.hpp>

#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/device_selector.hpp>
#include <sycl/sycl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;
using namespace ::testing;

namespace {

class ScopedBinaryRegistration {
public:
  explicit ScopedBinaryRegistration(llvm::ArrayRef<llvm::StringRef> KernelNames)
      : MBinary(sycl::unittest::createSYCLDeviceBinary(KernelNames)) {
    sycl::detail::ProgramAndKernelManager::getInstance().registerFatBin(
        MBinary.data(), MBinary.size());
  }

  ~ScopedBinaryRegistration() {
    sycl::detail::ProgramAndKernelManager::getInstance().unregisterFatBin(
        MBinary.data(), MBinary.size());
  }

private:
  llvm::SmallString<0> MBinary;
};

class DeviceSelectorScoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    // In reality gpu and cpu devices relate to different platforms. These
    // tests don't have to follow this rule since selectors work with types.
    Platform = mock::createDummyHandle<ol_platform_handle_t>();
    Device1 = mock::createDummyHandleWithData<ol_device_handle_t>(
        reinterpret_cast<unsigned char *>(&Platform), sizeof(Platform));
    Device2 = mock::createDummyHandleWithData<ol_device_handle_t>(
        reinterpret_cast<unsigned char *>(&Platform), sizeof(Platform));

    EXPECT_CALL(Helper.Mock.get(), olIterateDevices(_, _))
        .WillRepeatedly([this](ol_device_iterate_cb_t Callback,
                               void *UserData) -> ol_result_t {
          std::ignore = Callback(Device1, UserData);
          std::ignore = Callback(Device2, UserData);
          return OL_SUCCESS;
        });

    EXPECT_CALL(Helper.Mock.get(),
                olGetDeviceInfo(_, OL_DEVICE_INFO_PLATFORM, _, _))
        .WillRepeatedly([this](ol_device_handle_t Device,
                               ol_device_info_t /*PropName*/, size_t PropSize,
                               void *PropValue) -> ol_result_t {
          *static_cast<ol_platform_handle_t *>(PropValue) = Platform;
          return OL_SUCCESS;
        });
  }

  void TearDown() override {
    mock::releaseDummyHandles(Platform, Device1, Device2);
  }

  unittests::UnittestsHelper Helper;
  ol_platform_handle_t Platform{};
  ol_device_handle_t Device1{};
  ol_device_handle_t Device2{};
};

TEST_F(DeviceSelectorScoreTest, CPUAndGPU) {
  EXPECT_CALL(Helper.Mock.get(), olGetDeviceInfo(_, OL_DEVICE_INFO_TYPE, _, _))
      .WillRepeatedly([this](ol_device_handle_t Device,
                             ol_device_info_t /*PropName*/, size_t PropSize,
                             void *PropValue) -> ol_result_t {
        if (Device == Device1)
          *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_GPU;
        else if (Device == Device2)
          *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_CPU;
        else
          return mock::getMockLiboffload().makeEmptyStrError(
              OL_ERRC_INVALID_NULL_HANDLE);

        return OL_SUCCESS;
      });

  auto Devices = sycl::device::get_devices();
  ASSERT_EQ(Devices.size(), 2u);

  // Device order is aligned with device iteration order
  ASSERT_TRUE(Devices[0].is_gpu());
  ASSERT_TRUE(Devices[1].is_cpu());
  auto &GPUDevice = Devices[0];
  auto &CPUDevice = Devices[1];

  EXPECT_GT(sycl::default_selector_v(GPUDevice),
            sycl::default_selector_v(CPUDevice));

  EXPECT_GT(sycl::gpu_selector_v(GPUDevice), sycl::gpu_selector_v(CPUDevice));
  EXPECT_GT(sycl::gpu_selector_v(GPUDevice), 0);
  EXPECT_LT(sycl::gpu_selector_v(CPUDevice), 0);

  EXPECT_GT(sycl::cpu_selector_v(CPUDevice), sycl::cpu_selector_v(GPUDevice));
  EXPECT_GT(sycl::cpu_selector_v(CPUDevice), 0);
  EXPECT_LT(sycl::cpu_selector_v(GPUDevice), 0);

  EXPECT_LT(sycl::accelerator_selector_v(GPUDevice), 0);
  EXPECT_LT(sycl::accelerator_selector_v(CPUDevice), 0);
}

TEST_F(DeviceSelectorScoreTest, TwoGpusOneCompatibleImage) {
  EXPECT_CALL(Helper.Mock.get(), olGetDeviceInfo(_, OL_DEVICE_INFO_TYPE, _, _))
      .WillRepeatedly([](ol_device_handle_t Device,
                         ol_device_info_t /*PropName*/, size_t PropSize,
                         void *PropValue) -> ol_result_t {
        *static_cast<ol_device_type_t *>(PropValue) = OL_DEVICE_TYPE_GPU;
        return OL_SUCCESS;
      });

  EXPECT_CALL(Helper.Mock.get(), olIsValidBinary(_, _, _, _))
      .WillRepeatedly([this](ol_device_handle_t Device,
                             const void * /*ProgData*/, size_t /*ProgDataSize*/,
                             bool *Valid) -> ol_result_t {
        *Valid = (Device == Device2);
        return OL_SUCCESS;
      });

  std::array<llvm::StringRef, 1> KernelNames = {"kernel"};
  ScopedBinaryRegistration Registration{KernelNames};

  auto Devices = sycl::device::get_devices();
  ASSERT_EQ(Devices.size(), 2u);

  // Device order is aligned with device iteration order
  ASSERT_EQ(sycl::detail::getSyclObjImpl(Devices[1])->getOLHandle(), Device2);
  auto &GPUDevice = Devices[0];
  auto &GPUDeviceWithImage = Devices[1];

  EXPECT_GT(sycl::default_selector_v(GPUDeviceWithImage),
            sycl::default_selector_v(GPUDevice));

  sycl::device DefaultDevice{sycl::default_selector_v};
  auto DeviceDefaultNative =
      sycl::detail::getSyclObjImpl(DefaultDevice)->getOLHandle();
  EXPECT_EQ(DeviceDefaultNative, Device2);
}

TEST(DeviceSelector, AspectSelector) {
  unittests::UnittestsHelper Helper;
  auto Devices = sycl::device::get_devices();
  ASSERT_FALSE(Devices.empty());

  const sycl::device &Dev = Devices.front();

  const std::vector<sycl::aspect> EmptyAspects{};
  const std::vector<sycl::aspect> RequireGpu{sycl::aspect::gpu};
  const std::vector<sycl::aspect> DenyGpu{sycl::aspect::gpu};

  auto FallbackSelector = sycl::aspect_selector(EmptyAspects, EmptyAspects);
  EXPECT_EQ(FallbackSelector(Dev), sycl::default_selector_v(Dev));

  auto RequireGpuSelector = sycl::aspect_selector(RequireGpu, EmptyAspects);
  EXPECT_GT(RequireGpuSelector(Dev), 0);

  auto DenyGpuSelector = sycl::aspect_selector(EmptyAspects, DenyGpu);
  EXPECT_LT(DenyGpuSelector(Dev), 0);
}

} // namespace
