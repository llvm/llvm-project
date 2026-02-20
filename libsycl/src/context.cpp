//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>

#include <detail/context_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

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
