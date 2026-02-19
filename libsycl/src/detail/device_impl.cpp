//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

bool DeviceImpl::has(aspect Aspect) const {
  switch (Aspect) {
  case (aspect::cpu):
    return isCPU();
  case (aspect::gpu):
    return isGPU();
  case (aspect::accelerator):
    return isAccelerator();
  case (aspect::custom):
    return false;
  case (aspect::emulated):
    return false;
  case (aspect::host_debuggable):
    return false;
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

backend DeviceImpl::getBackend() const { return MPlatform.getBackend(); }

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
