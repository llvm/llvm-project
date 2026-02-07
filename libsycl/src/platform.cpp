//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/platform.hpp>

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

backend platform::get_backend() const noexcept { return impl->getBackend(); }

std::vector<platform> platform::get_platforms() {
  auto &PlatformImpls = detail::PlatformImpl::getPlatforms();
  std::vector<platform> Platforms;
  Platforms.reserve(PlatformImpls.size());
  for (auto &PlatformImpl : PlatformImpls) {
    Platforms.emplace_back(
        detail::createSyclObjFromImpl<platform>(*PlatformImpl.get()));
  }
  return Platforms;
}

std::vector<device> platform::get_devices(info::device_type DeviceType) const {
  std::vector<device> Devices;
  impl->iterateDevices(DeviceType, [&Devices](detail::DeviceImpl *DevImpl) {
    assert(DevImpl && "Device impl can't be nullptr");
    Devices.push_back(detail::createSyclObjFromImpl<device>(*DevImpl));
  });

  return Devices;
}

bool platform::has(aspect Aspect) const { return impl->has(Aspect); }

template <typename Param>
detail::is_platform_info_desc_t<Param> platform::get_info() const {
  return impl->getInfo<Param>();
}

#define _LIBSYCL_EXPORT_GET_INFO(Desc)                                         \
  template _LIBSYCL_EXPORT                                                     \
      detail::is_platform_info_desc_t<info::platform::Desc>                    \
      platform::get_info<info::platform::Desc>() const;
_LIBSYCL_EXPORT_GET_INFO(version)
_LIBSYCL_EXPORT_GET_INFO(name)
_LIBSYCL_EXPORT_GET_INFO(vendor)
#undef _LIBSYCL_EXPORT_GET_INFO

_LIBSYCL_END_NAMESPACE_SYCL
