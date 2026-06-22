//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>

#include <detail/program_manager.hpp>

#include <sycl/__impl/exception.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <string>

#include <llvm/ADT/SmallVector.h>

using namespace sycl;

using namespace ::testing;

struct MockProgramAndKernelManager : public detail::ProgramAndKernelManager {
  using detail::ProgramAndKernelManager::MDeviceImageManagers;
  using detail::ProgramAndKernelManager::MDeviceKernelInfoMap;
};

TEST(ProgramAndKernelManager, CheckUnsupportedVersionOfFatbin) {
  std::array<llvm::StringRef, 1> KernelNames = {"kernel"};
  llvm::SmallString<0> Binary =
      sycl::unittest::createSYCLDeviceBinary(KernelNames);

  llvm::MemoryBufferRef MBR(
      llvm::StringRef(static_cast<const char *>(Binary.data()), Binary.size()),
      /*Identifier=*/"");
  auto Header = llvm::object::OffloadBinary::extractHeader(MBR);
  ASSERT_TRUE(static_cast<const bool>(Header));
  // extractHeader returns const reference to the header, hacking it.
  auto *ModifiableHeader =
      const_cast<llvm::object::OffloadBinary::Header *>(Header.get());
  ModifiableHeader->Version = llvm::object::OffloadBinary::Version + 1;

  MockProgramAndKernelManager Manager;
  EXPECT_THAT([&]() { Manager.registerFatBin(Binary.data(), Binary.size()); },
              Throws<sycl::exception>(AllOf(
                  Property(&sycl::exception::what,
                           HasSubstr("Failed to parse OffloadBinary")),
                  Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));

  ModifiableHeader->Version = llvm::object::OffloadBinary::Version;
  EXPECT_NO_THROW(Manager.registerFatBin(Binary.data(), Binary.size()));
  EXPECT_NO_THROW(Manager.unregisterFatBin(Binary.data(), Binary.size()));
}

TEST(ProgramAndKernelManager, CheckUnsupportedVersionOfImage) {
  std::array<llvm::StringRef, 1> KernelNames = {"kernel"};
  llvm::SmallString<0> IncompatibleImageBinary =
      sycl::unittest::createSYCLDeviceBinary(KernelNames,
                                             llvm::object::IMG_Bitcode);

  MockProgramAndKernelManager Manager;

  EXPECT_THAT(
      [&]() {
        Manager.registerFatBin(IncompatibleImageBinary.data(),
                               IncompatibleImageBinary.size());
      },
      Throws<sycl::exception>(
          AllOf(Property(&sycl::exception::what,
                         HasSubstr("Incompatible device image")),
                Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));

  llvm::SmallString<0> CompatibleImageBinary =
      sycl::unittest::createSYCLDeviceBinary(KernelNames,
                                             llvm::object::IMG_SPIRV);
  EXPECT_NO_THROW(Manager.registerFatBin(CompatibleImageBinary.data(),
                                         CompatibleImageBinary.size()));
  EXPECT_NO_THROW(Manager.unregisterFatBin(CompatibleImageBinary.data(),
                                           CompatibleImageBinary.size()));
}

TEST(ProgramAndKernelManager, CheckRegisterAndUnregister) {
  std::array<std::string, 3> KernelNames = {"kernel1.1", "kernel1.2",
                                            "kernel2.1"};
  std::array<llvm::StringRef, 2> Image1Kernels = {KernelNames[0],
                                                  KernelNames[1]};
  std::array<llvm::StringRef, 1> Image2Kernels = {KernelNames[2]};

  std::array<llvm::SmallString<0>, 2> Symbols;
  llvm::offloading::sycl::writeSymbolTable(Image1Kernels, Symbols[0]);
  llvm::offloading::sycl::writeSymbolTable(Image2Kernels, Symbols[1]);

  llvm::SmallVector<llvm::object::OffloadBinary::OffloadingImage, 2> Images;
  Images.push_back(sycl::unittest::createSYCLImage(Symbols[0]));
  Images.push_back(sycl::unittest::createSYCLImage(Symbols[1]));

  llvm::SmallString<0> Binary = llvm::object::OffloadBinary::write(Images);

  MockProgramAndKernelManager Manager;
  EXPECT_NO_THROW(Manager.registerFatBin(Binary.data(), Binary.size()));

  ASSERT_THAT(Manager.MDeviceImageManagers, SizeIs(1));
  auto ImagesIt = Manager.MDeviceImageManagers.find(Binary.data());
  ASSERT_NE(ImagesIt, Manager.MDeviceImageManagers.end());
  ASSERT_THAT(ImagesIt->second, SizeIs(2));

  ASSERT_THAT(Manager.MDeviceKernelInfoMap, SizeIs(KernelNames.size()));
  for (auto &[Name, KernelInfo] : Manager.MDeviceKernelInfoMap) {
    // Check all name related fields
    EXPECT_EQ(Name, KernelInfo.getName());
    EXPECT_THAT(KernelNames, Contains(Name));

    // Check device image ref correctness in kernel info.
    auto &DevImage = KernelInfo.getDeviceImage();
    uint64_t ImageIndex = DevImage.getOffloadBinary().getIndex();
    if (Name.find("kernel1") != Name.npos)
      EXPECT_EQ(ImageIndex, 0u);
    else
      EXPECT_EQ(ImageIndex, 1u);
  }

  EXPECT_NO_THROW(Manager.unregisterFatBin(Binary.data(), Binary.size()));
  EXPECT_THAT(Manager.MDeviceImageManagers, IsEmpty());
  EXPECT_THAT(Manager.MDeviceKernelInfoMap, IsEmpty());
}
