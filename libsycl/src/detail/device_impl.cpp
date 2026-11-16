//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

#include <optional>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

// Returns the liboffload boolean query reporting support of Aspect, or
// std::nullopt if liboffload has no such query.
static std::optional<ol_device_info_t> mapAspectToOLInfo(aspect Aspect) {
  switch (Aspect) {
  case (aspect::fp16):
    return OL_DEVICE_INFO_HALF_FP_SUPPORT;
  case (aspect::fp64):
    return OL_DEVICE_INFO_DOUBLE_FP_SUPPORT;
  default:
    return std::nullopt;
  }
}

bool DeviceImpl::has(aspect Aspect) const {
  if (std::optional<ol_device_info_t> Info = mapAspectToOLInfo(Aspect)) {
    bool Supported = false;
    callAndThrow(olGetDeviceInfo, MOffloadDevice, *Info, sizeof(Supported),
                 &Supported);
    return Supported;
  }

  switch (Aspect) {
  case (aspect::cpu):
    return isCPU();
  case (aspect::gpu):
    return isGPU();
  case (aspect::accelerator):
    return isAccelerator();
  case (aspect::custom):
  case (aspect::emulated):
  case (aspect::host_debuggable):
    return false;
  case (aspect::usm_device_allocations):
  case (aspect::usm_host_allocations):
  case (aspect::usm_shared_allocations):
    // liboffload works with USM only and has no query to check support. We
    // assume that USM is always supported.
    return true;
  default:
    // Other aspects are not implemented yet
    return false;
  }
}

info::device_type DeviceImpl::getDeviceType() const {
  return getInfo<info::device::device_type>();
}

bool DeviceImpl::isCPU() const {
  return getDeviceType() == info::device_type::cpu;
}

bool DeviceImpl::isGPU() const {
  return getDeviceType() == info::device_type::gpu;
}

bool DeviceImpl::isAccelerator() const {
  return getDeviceType() == info::device_type::accelerator;
}

backend DeviceImpl::getBackend() const noexcept {
  return MPlatform.getBackend();
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
