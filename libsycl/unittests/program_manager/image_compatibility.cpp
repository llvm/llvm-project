//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for matching a registered device image against a device: only the
/// architecture component of the image's target triple takes part in it.
///
//===----------------------------------------------------------------------===//

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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace sycl;
using namespace ::testing;

namespace {

/// Creates a context over a single device, bypassing sycl::context, which
/// exposes a platform's default context only.
std::shared_ptr<detail::ContextImpl> createContext(const device &Device) {
  std::vector<detail::DeviceImpl *> Devices = {detail::getSyclObjImpl(Device)};
  return detail::ContextImpl::create(
      std::move(Devices), detail::defaultAsyncHandler, property_list{});
}

/// Allows the liboffload calls that these tests do not assert on, so that
/// context and program lifetime does not produce uninteresting call warnings.
void allowContextAndProgramLifetimeCalls(mock::MockLiboffload &Mock) {
  EXPECT_CALL(Mock, olCreateContext(_, _, _)).Times(AnyNumber());
  EXPECT_CALL(Mock, olDestroyContext(_)).Times(AnyNumber());
  EXPECT_CALL(Mock, olCreateProgram(_, _, _, _, _)).Times(AnyNumber());
  EXPECT_CALL(Mock, olDestroyProgram(_)).Times(AnyNumber());
  EXPECT_CALL(Mock, olGetSymbol(_, _, _, _)).Times(AnyNumber());
}

ol_symbol_handle_t
getKernel(const std::shared_ptr<detail::ContextImpl> &Context,
          const device &Device, std::string_view KernelName) {
  detail::ProgramAndKernelManager &Manager =
      detail::ProgramAndKernelManager::getInstance();
  return Manager.getOrCreateKernel(Manager.getDeviceKernelInfo(KernelName),
                                   Context, *detail::getSyclObjImpl(Device));
}

} // namespace

// An image whose triple carries an extra environment component is still usable.
TEST(ImageCompatibility, ArchitectureAloneSelectsImage) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  sycl::unittests::ScopedKernelRegistration Registration(
      KernelName, "spirv64-unknown-unknown-elf");

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);

  EXPECT_NE(getKernel(Context, Device, KernelName), nullptr);
}

// An image built for another architecture must not be handed to the device.
TEST(ImageCompatibility, ImageForAnotherArchitectureIsRejected) {
  mock::MockWrapper Mock;
  allowContextAndProgramLifetimeCalls(Mock.get());

  const std::string KernelName = "kernel";
  sycl::unittests::ScopedKernelRegistration Registration(
      KernelName, "spirv32-unknown-unknown");

  const device Device;
  std::shared_ptr<detail::ContextImpl> Context = createContext(Device);

  EXPECT_THAT([&]() { getKernel(Context, Device, KernelName); },
              Throws<sycl::exception>(AllOf(
                  Property(&sycl::exception::what,
                           HasSubstr("No compatible image for " + KernelName)),
                  Property(&sycl::exception::code, Eq(sycl::errc::runtime)))));
}
