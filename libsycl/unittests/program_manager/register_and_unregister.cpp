//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mock/device_images.hpp>
#include <mock/helpers.hpp>

#include <detail/device_image_wrapper.hpp>
#include <detail/device_kernel_info.hpp>
#include <detail/program_manager.hpp>

#include <sycl/__impl/exception.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace sycl;

using namespace ::testing;

struct MockProgramAndKernelManager : public detail::ProgramAndKernelManager {
  using detail::ProgramAndKernelManager::MDeviceImageManagers;
  using detail::ProgramAndKernelManager::MDeviceKernelInfoMap;
};

TEST(ProgramAndKernelManager, CheckUnsupportedVersionOfFatbin) {
  sycl::detail::__sycl_tgt_bin_desc DeviceImages =
      sycl::unittest::GenericDeviceImages;
  DeviceImages.Version = 3;

  MockProgramAndKernelManager Manager;
  EXPECT_THAT(
      [&]() { Manager.registerFatBin(&DeviceImages); },
      Throws<sycl::exception>(AllOf(
          Property(
              &sycl::exception::what,
              HasSubstr("Incompatible version of device images descriptor")),
          Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));

  DeviceImages.Version = 1;
  EXPECT_NO_THROW(Manager.registerFatBin(&DeviceImages));
}

TEST(ProgramAndKernelManager, CheckUnsupportedVersionOfImage) {
  std::array<sycl::detail::__sycl_tgt_device_image, 1> DevImage{
      sycl::unittest::GenericDeviceImage};
  DevImage[0].Version = 2;
  sycl::detail::__sycl_tgt_bin_desc DeviceImages =
      sycl::unittest::GenericDeviceImages;
  DeviceImages.NumDeviceBinaries = DevImage.size();
  DeviceImages.DeviceImages = DevImage.data();

  MockProgramAndKernelManager Manager;

  EXPECT_THAT([&]() { Manager.registerFatBin(&DeviceImages); },
              Throws<sycl::exception>(AllOf(
                  Property(&sycl::exception::what,
                           HasSubstr("Incompatible device image")),
                  Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));

  DevImage[0].Version = 3;
  EXPECT_NO_THROW(Manager.registerFatBin(&DeviceImages));
}

TEST(ProgramAndKernelManager, CheckRegisterAndUnregister) {
  std::array<std::string, 3> KernelNames = {"kernel1.1", "kernel1.2",
                                            "kernel2.1"};
  std::array<llvm::offloading::EntryTy, 2> Entries1 = {
      sycl::unittest::GenericEntry, sycl::unittest::GenericEntry};
  Entries1[0].SymbolName = KernelNames[0].data();
  Entries1[0].Size = KernelNames[0].size();
  Entries1[1].SymbolName = KernelNames[1].data();
  Entries1[1].Size = KernelNames[1].size();

  std::array<llvm::offloading::EntryTy, 1> Entries2 = {
      sycl::unittest::GenericEntry};
  Entries2[0].SymbolName = KernelNames[2].data();
  Entries2[0].Size = KernelNames[2].size();

  std::array<sycl::detail::__sycl_tgt_device_image, 2> DevImages = {
      sycl::unittest::GenericDeviceImage, sycl::unittest::GenericDeviceImage};
  DevImages[0].EntriesBegin = Entries1.begin();
  DevImages[0].EntriesEnd = Entries1.end();
  DevImages[1].EntriesBegin = Entries2.begin();
  DevImages[1].EntriesEnd = Entries2.end();

  sycl::detail::__sycl_tgt_bin_desc DeviceImagesDesc =
      sycl::unittest::GenericDeviceImages;
  DeviceImagesDesc.NumDeviceBinaries = DevImages.size();
  DeviceImagesDesc.DeviceImages = DevImages.data();

  MockProgramAndKernelManager Manager;
  EXPECT_NO_THROW(Manager.registerFatBin(&DeviceImagesDesc));

  ASSERT_THAT(Manager.MDeviceImageManagers, SizeIs(DevImages.size()));
  for (auto &Image : DevImages) {
    // contains comes with c++20, so just check count == 1 here.
    EXPECT_EQ(Manager.MDeviceImageManagers.count(&Image), 1u);
  }

  ASSERT_THAT(Manager.MDeviceKernelInfoMap, SizeIs(KernelNames.size()));
  for (auto &[Name, KernelInfo] : Manager.MDeviceKernelInfoMap) {
    // Check all name related fields
    EXPECT_EQ(Name, KernelInfo.getName());
    EXPECT_THAT(KernelNames, Contains(Name));

    // Check device image ref correctness in kernel info.
    auto &DevImage = KernelInfo.getDeviceImage();
    if (Name.find("kernel1") != Name.npos)
      EXPECT_EQ(&DevImage.getRawData(), &DevImages[0]);
    else
      EXPECT_EQ(&DevImage.getRawData(), &DevImages[1]);
  }

  // Unregister the first image with 2 kernels.
  sycl::detail::__sycl_tgt_bin_desc SingleImageDesc =
      sycl::unittest::GenericDeviceImages;
  SingleImageDesc.NumDeviceBinaries = 1;
  SingleImageDesc.DeviceImages = DevImages.data();
  EXPECT_NO_THROW(Manager.unregisterFatBin(&SingleImageDesc));

  ASSERT_THAT(Manager.MDeviceImageManagers, SizeIs(1));
  EXPECT_EQ(Manager.MDeviceImageManagers.begin()->first, &DevImages[1]);
  ASSERT_THAT(Manager.MDeviceKernelInfoMap, SizeIs(1));
  EXPECT_EQ(KernelNames[2].data(),
            Manager.MDeviceKernelInfoMap.begin()->second.getName());
}
