//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

// c++20 designated initializers would be helpful.
constexpr llvm::offloading::EntryTy GGenericEntryTy = {
    /// Reserved bytes used to detect an older version of the struct, always
    /// zero.
    0,
    /// The current version of the struct for runtime forward compatibility.
    1,
    /// The expected consumer of this entry, e.g. CUDA or OpenMP.
    llvm::object::OFK_SYCL,
    /// Flags associated with the global.
    0,
    /// The address of the global to be registered by the runtime.
    nullptr,
    /// The name of the symbol in the device image.
    nullptr,
    /// The number of bytes the symbol takes.
    0,
    /// Extra generic data used to register this entry.
    0,
    /// An extra pointer, usually null.
    nullptr};

constexpr sycl::detail::__sycl_tgt_device_image GGenericDeviceImage = {
    // Version
    3,
    // OffloadKind
    llvm::object::OFK_SYCL,
    // ImageFormat
    llvm::object::IMG_SPIRV,
    // TripleString
    "spirv64-unknown-unknown",
    // CompileOptions
    "",
    // LinkOptions
    "",
    // ImageStart
    nullptr,
    // ImageEnd
    nullptr,
    // EntriesBegin
    nullptr,
    // EntriesEnd
    nullptr,
    // PropertiesBegin
    nullptr,
    // PropertiesEnd
    nullptr};

constexpr sycl::detail::__sycl_tgt_bin_desc GGenericDeviceImages = {
    // Version.
    1,
    // Num binaries
    0,
    /// Device binaries data.
    nullptr,
    // HostEntriesBegin.
    nullptr,
    // HostEntriesEnd.
    nullptr};

TEST(ProgramAndKernelManager, CheckUnsupportedVersionOfFatbin) {
  sycl::detail::__sycl_tgt_bin_desc DeviceImages = GGenericDeviceImages;
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
      GGenericDeviceImage};
  DevImage[0].Version = 2;
  sycl::detail::__sycl_tgt_bin_desc DeviceImages = GGenericDeviceImages;
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
  std::array<llvm::offloading::EntryTy, 2> Entries1 = {GGenericEntryTy,
                                                       GGenericEntryTy};
  Entries1[0].SymbolName = KernelNames[0].data();
  Entries1[0].Size = KernelNames[0].size();
  Entries1[1].SymbolName = KernelNames[1].data();
  Entries1[1].Size = KernelNames[1].size();

  std::array<llvm::offloading::EntryTy, 1> Entries2 = {GGenericEntryTy};
  Entries2[0].SymbolName = KernelNames[2].data();
  Entries2[0].Size = KernelNames[2].size();

  std::array<sycl::detail::__sycl_tgt_device_image, 2> DevImages = {
      GGenericDeviceImage, GGenericDeviceImage};
  DevImages[0].EntriesBegin = Entries1.begin();
  DevImages[0].EntriesEnd = Entries1.end();
  DevImages[1].EntriesBegin = Entries2.begin();
  DevImages[1].EntriesEnd = Entries2.end();

  sycl::detail::__sycl_tgt_bin_desc DeviceImagesDesc = GGenericDeviceImages;
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
  sycl::detail::__sycl_tgt_bin_desc SingleImageDesc = GGenericDeviceImages;
  SingleImageDesc.NumDeviceBinaries = 1;
  SingleImageDesc.DeviceImages = DevImages.data();
  EXPECT_NO_THROW(Manager.unregisterFatBin(&SingleImageDesc));

  ASSERT_THAT(Manager.MDeviceImageManagers, SizeIs(1));
  EXPECT_EQ(Manager.MDeviceImageManagers.begin()->first, &DevImages[1]);
  ASSERT_THAT(Manager.MDeviceKernelInfoMap, SizeIs(1));
  EXPECT_EQ(KernelNames[2].data(),
            Manager.MDeviceKernelInfoMap.begin()->second.getName());
}
