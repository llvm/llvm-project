//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for the per-context program cache: a program is identified by the
/// (context, device, image) triple, and it must be destroyed before both the
/// context it belongs to and the image it was created from.
///
//===----------------------------------------------------------------------===//

#include <common/device_images.hpp>
#include <common/scoped_binary_registration.hpp>
#include <mock/helpers.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/program_manager.hpp>

#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/property_list.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <llvm/ADT/SmallVector.h>

using namespace sycl;
using namespace ::testing;

namespace {

/// Creates a context over a single device, bypassing sycl::context: SYCL 2020
/// only exposes a platform's default context here, and these tests need several
/// distinct contexts over the same device.
std::shared_ptr<detail::ContextImpl> createContext(const device &Device) {
  std::vector<detail::DeviceImpl *> Devices = {detail::getSyclObjImpl(Device)};
  return detail::ContextImpl::create(
      std::move(Devices), detail::defaultAsyncHandler, property_list{});
}

/// Allows the liboffload calls that these tests do not assert on, so that
/// context and program teardown does not produce uninteresting call warnings.
void allowContextAndProgramLifetimeCalls(mock::MockLiboffload &Mock) {
  EXPECT_CALL(Mock, olCreateContext(_, _, _)).Times(AnyNumber());
  EXPECT_CALL(Mock, olDestroyContext(_)).Times(AnyNumber());
  EXPECT_CALL(Mock, olCreateProgram(_, _, _, _, _)).Times(AnyNumber());
  EXPECT_CALL(Mock, olDestroyProgram(_)).Times(AnyNumber());
  EXPECT_CALL(Mock, olGetSymbol(_, _, _, _)).Times(AnyNumber());
}

detail::DeviceKernelInfo &getKernelInfo(std::string_view KernelName) {
  return detail::ProgramAndKernelManager::getInstance().getDeviceKernelInfo(
      KernelName);
}

ol_symbol_handle_t
getKernel(const std::shared_ptr<detail::ContextImpl> &Context,
          const device &Device, std::string_view KernelName) {
  return detail::ProgramAndKernelManager::getInstance().getOrCreateKernel(
      getKernelInfo(KernelName), Context, *detail::getSyclObjImpl(Device));
}

} // namespace

// A program belongs to the context it was created in, so two contexts over the
// same device must not share one.
TEST(ProgramCache, ProgramIsCreatedPerContext) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  sycl::unittests::ScopedKernelRegistration Registration(KernelName);

  const device Device;
  std::shared_ptr<detail::ContextImpl> FirstContext = createContext(Device);
  std::shared_ptr<detail::ContextImpl> SecondContext = createContext(Device);

  EXPECT_CALL(Mock.get(), olCreateProgram(_, _, _, _, _)).Times(2);

  ol_symbol_handle_t FirstKernel = getKernel(FirstContext, Device, KernelName);
  ol_symbol_handle_t SecondKernel =
      getKernel(SecondContext, Device, KernelName);
  EXPECT_NE(FirstKernel, nullptr);
  EXPECT_NE(SecondKernel, nullptr);
  EXPECT_NE(FirstKernel, SecondKernel);
}

// A repeated request within the same context must be served from the cache.
TEST(ProgramCache, ProgramAndKernelAreCached) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  sycl::unittests::ScopedKernelRegistration Registration(KernelName);

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);

  EXPECT_CALL(Mock.get(), olCreateProgram(_, _, _, _, _)).Times(1);
  EXPECT_CALL(Mock.get(), olGetSymbol(_, _, _, _)).Times(1);

  ol_symbol_handle_t FirstKernel = getKernel(Context, Device, KernelName);
  ol_symbol_handle_t SecondKernel = getKernel(Context, Device, KernelName);
  EXPECT_EQ(FirstKernel, SecondKernel);
}

// Two images registered for the same device need two programs. The cache used
// to be keyed by device alone, which handed out the first image's program for
// kernels of the second one.
TEST(ProgramCache, ProgramIsCreatedPerDeviceImage) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  std::array<std::string, 2> KernelNames = {"image1kernel", "image2kernel"};
  std::array<llvm::StringRef, 1> Image1Kernels = {KernelNames[0]};
  std::array<llvm::StringRef, 1> Image2Kernels = {KernelNames[1]};

  std::array<llvm::SmallString<0>, 2> Symbols;
  llvm::offloading::sycl::writeSymbolTable(Image1Kernels, Symbols[0]);
  llvm::offloading::sycl::writeSymbolTable(Image2Kernels, Symbols[1]);

  llvm::SmallVector<llvm::object::OffloadBinary::OffloadingImage, 2> Images;
  Images.push_back(sycl::unittests::createSYCLImage(Symbols[0]));
  Images.push_back(sycl::unittests::createSYCLImage(Symbols[1]));
  llvm::SmallString<0> Binary = llvm::object::OffloadBinary::write(Images);

  detail::ProgramAndKernelManager &Manager =
      detail::ProgramAndKernelManager::getInstance();
  Manager.registerFatBin(Binary.data(), Binary.size());

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);

  EXPECT_CALL(Mock.get(), olCreateProgram(_, _, _, _, _)).Times(2);

  ol_symbol_handle_t FirstKernel = getKernel(Context, Device, KernelNames[0]);
  ol_symbol_handle_t SecondKernel = getKernel(Context, Device, KernelNames[1]);
  EXPECT_NE(FirstKernel, SecondKernel);

  Manager.unregisterFatBin(Binary.data(), Binary.size());
}

// liboffload does not reference-count contexts: a program tied to a context
// that has already been destroyed is in an undefined state, so olDestroyProgram
// must come first.
TEST(ProgramCache, ProgramsAreDestroyedBeforeContext) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  sycl::unittests::ScopedKernelRegistration Registration(KernelName);

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);
  EXPECT_NE(getKernel(Context, Device, KernelName), nullptr);

  {
    InSequence Sequence;
    EXPECT_CALL(Mock.get(), olDestroyProgram(_)).Times(1);
    EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);
  }

  Context.reset();
}

// Programs are created from the image's memory and cache kernel names that
// point into it, so unregistering the image must release them even though the
// context that owns them stays alive.
TEST(ProgramCache, ProgramsAreDestroyedOnImageUnregistration) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  std::array<llvm::StringRef, 1> KernelNames = {KernelName};
  llvm::SmallString<0> Binary =
      sycl::unittests::createSYCLDeviceBinary(KernelNames);

  detail::ProgramAndKernelManager &Manager =
      detail::ProgramAndKernelManager::getInstance();
  Manager.registerFatBin(Binary.data(), Binary.size());

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);
  EXPECT_NE(getKernel(Context, Device, KernelName), nullptr);

  EXPECT_CALL(Mock.get(), olDestroyProgram(_)).Times(1);
  Manager.unregisterFatBin(Binary.data(), Binary.size());
  // Qualified: the local Mock variable shadows ::testing::Mock here.
  ::testing::Mock::VerifyAndClearExpectations(&Mock.get());

  // Nothing is left for the context to release.
  EXPECT_CALL(Mock.get(), olDestroyProgram(_)).Times(0);
  EXPECT_CALL(Mock.get(), olDestroyContext(_)).Times(1);
  Context.reset();
}
