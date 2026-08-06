//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/context.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/exception.hpp>

#include <detail/context_impl.hpp>
#include <detail/platform_impl.hpp>

#include <algorithm>
#include <cassert>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

context::context(const property_list &propList) : context(device{}, propList) {}

context::context(async_handler asyncHandler, const property_list &propList)
    : context(device{}, asyncHandler, propList) {}

context::context(const device &dev, const property_list &propList)
    : context(std::vector<device>{dev}, propList) {}

context::context(const device &dev, async_handler asyncHandler,
                 const property_list &propList)
    : context(std::vector<device>{dev}, asyncHandler, propList) {}

context::context(const platform &plt, const property_list &propList)
    : context(plt.get_devices(), propList) {}

context::context(const platform &plt, async_handler asyncHandler,
                 const property_list &propList)
    : context(plt.get_devices(), asyncHandler, propList) {}

context::context(const std::vector<device> &deviceList,
                 const property_list &propList)
    : context(deviceList, detail::defaultAsyncHandler, propList) {}

context::context(const std::vector<device> &deviceList,
                 async_handler asyncHandler, const property_list &propList)
    : impl() {
  if (deviceList.empty()) {
    throw exception(make_error_code(errc::invalid),
                    "Device list must not be empty");
  }

  const auto &platform = deviceList[0].get_platform();
  if (std::any_of(deviceList.begin(), deviceList.end(),
                  [&platform](const device &dev) {
                    return dev.get_platform() != platform;
                  })) {
    throw exception(make_error_code(errc::invalid),
                    "All devices must be associated with the same platform");
  }

  std::vector<detail::DeviceImpl *> deviceImpls;
  deviceImpls.reserve(deviceList.size());
  for (const device &dev : deviceList) {
    deviceImpls.push_back(detail::ImplUtils::getSyclObjImpl(dev));
  }

  impl = detail::ContextImpl::create(std::move(deviceImpls), asyncHandler,
                                     propList);
  assert(impl && "ContextImpl creation failed");
}

backend context::get_backend() const noexcept { return impl->getBackend(); }

platform context::get_platform() const {
  return detail::createSyclObjFromImpl<platform>(impl->getPlatformImpl());
}

std::vector<device> context::get_devices() const {
  std::vector<device> Devices;

  impl->iterateDevices([&Devices](detail::DeviceImpl *DevImpl) {
    assert(DevImpl && "Device impl can't be nullptr");
    Devices.push_back(detail::createSyclObjFromImpl<device>(*DevImpl));
  });

  return Devices;
}

_LIBSYCL_END_NAMESPACE_SYCL
