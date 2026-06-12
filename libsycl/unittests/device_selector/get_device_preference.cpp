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

  for (const auto &Dev : Devices) {
    if (Dev.is_gpu()) {
      EXPECT_EQ(sycl::default_selector_v(Dev), 550);
      EXPECT_EQ(sycl::gpu_selector_v(Dev), 1050);
      EXPECT_EQ(sycl::cpu_selector_v(Dev), -1);
      EXPECT_EQ(sycl::accelerator_selector_v(Dev), -1);
    } else if (Dev.is_cpu()) {
      EXPECT_EQ(sycl::default_selector_v(Dev), 350);
      EXPECT_EQ(sycl::gpu_selector_v(Dev), -1);
      EXPECT_EQ(sycl::cpu_selector_v(Dev), 1050);
      EXPECT_EQ(sycl::accelerator_selector_v(Dev), -1);
    } else
      FAIL() << "Unexpected device type";
  }
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

  auto DeviceNative = sycl::detail::getSyclObjImpl(Devices[0])->getOLHandle();
  int Score = sycl::default_selector_v(Devices[0]);
  if (DeviceNative == Device1)
    EXPECT_EQ(Score, 550);
  else if (DeviceNative == Device2)
    EXPECT_EQ(Score, 1550);
  else
    FAIL() << "Unexpected device handle: ";

  sycl::device DefaultDevice{sycl::default_selector_v};
  auto DeviceDefaultNative =
      sycl::detail::getSyclObjImpl(DefaultDevice)->getOLHandle();
  EXPECT_EQ(DeviceDefaultNative, Device2);
}

TEST(DeviceSelector, AspectSelector) {
  auto Devices = sycl::device::get_devices();
  ASSERT_FALSE(Devices.empty());

  const sycl::device &Dev = Devices.front();

  const std::vector<sycl::aspect> EmptyAspects{};
  const std::vector<sycl::aspect> RequireGpu{sycl::aspect::gpu};
  const std::vector<sycl::aspect> DenyGpu{sycl::aspect::gpu};

  auto FallbackSelector = sycl::aspect_selector(EmptyAspects, EmptyAspects);
  EXPECT_EQ(FallbackSelector(Dev), sycl::default_selector_v(Dev));

  auto RequireGpuSelector = sycl::aspect_selector(RequireGpu, EmptyAspects);
  EXPECT_EQ(RequireGpuSelector(Dev), 1050);

  auto DenyGpuSelector = sycl::aspect_selector(EmptyAspects, DenyGpu);
  EXPECT_EQ(DenyGpuSelector(Dev), -1);
}

} // namespace
