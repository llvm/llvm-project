//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/__impl/device.hpp>

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

#include <cassert>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

bool device::is_cpu() const { return impl->isCPU(); }

bool device::is_gpu() const { return impl->isGPU(); }

bool device::is_accelerator() const { return impl->isAccelerator(); }

platform device::get_platform() const {
  return detail::createSyclObjFromImpl<platform>(impl->getPlatformImpl());
}

backend device::get_backend() const noexcept { return impl->getBackend(); }

std::vector<device> device::get_devices(info::device_type type) {
  std::vector<device> Devices;

  // Not calling platform::get_devices to avoid multiple vector packing
  for (const auto &Impl : detail::PlatformImpl::getPlatforms()) {
    assert(Impl && "PlatformImpl can not be nullptr");
    Impl->iterateDevices(type, [&Devices](detail::DeviceImpl *DevImpl) {
      assert(DevImpl && "Device impl can't be nullptr");
      Devices.push_back(detail::createSyclObjFromImpl<device>(*DevImpl));
    });
  }

  return Devices;
}

template <info::partition_property Prop>
std::vector<device> device::create_sub_devices(std::size_t /*count*/) const {
  throw exception(make_error_code(errc::feature_not_supported),
                  "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_equally>(
    std::size_t count) const;

template <info::partition_property Prop>
std::vector<device>
device::create_sub_devices(const std::vector<std::size_t> & /*counts*/) const {
  throw exception(make_error_code(errc::feature_not_supported),
                  "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device>
device::create_sub_devices<info::partition_property::partition_by_counts>(
    const std::vector<std::size_t> &counts) const;

template <info::partition_property Prop>
std::vector<device> device::create_sub_devices(
    info::partition_affinity_domain /*affinityDomain*/) const {
  throw exception(make_error_code(errc::feature_not_supported),
                  "Partitioning is not supported.");
}

template _LIBSYCL_EXPORT std::vector<device> device::create_sub_devices<
    info::partition_property::partition_by_affinity_domain>(
    info::partition_affinity_domain affinityDomain) const;

bool device::has(aspect asp) const { return impl->has(asp); }

template <typename Param>
detail::is_device_info_desc_t<Param> device::get_info() const {
  return impl->getInfo<Param>();
}

template <>
_LIBSYCL_EXPORT detail::is_device_info_desc_t<info::device::platform>
device::get_info<info::device::platform>() const {
  static_assert(
      std::is_same_v<info::device::platform::return_type, sycl::platform>);
  return get_platform();
}

#define _LIBSYCL_EXPORT_GET_INFO(Desc)                                         \
  template _LIBSYCL_EXPORT detail::is_device_info_desc_t<info::device::Desc>   \
  device::get_info<info::device::Desc>() const;
_LIBSYCL_EXPORT_GET_INFO(device_type)
_LIBSYCL_EXPORT_GET_INFO(name)
_LIBSYCL_EXPORT_GET_INFO(vendor)
_LIBSYCL_EXPORT_GET_INFO(driver_version)
#undef _LIBSYCL_EXPORT_GET_INFO

_LIBSYCL_END_NAMESPACE_SYCL
