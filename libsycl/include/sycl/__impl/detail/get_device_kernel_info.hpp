//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper function to query kernel info that is uniform
/// between different submissions of the same kernel.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_GET_DEVICE_KERNEL_INFO_HPP
#define _LIBSYCL___IMPL_DETAIL_GET_DEVICE_KERNEL_INFO_HPP

#include <sycl/__impl/detail/config.hpp>

#include <string_view>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

class DeviceKernelInfo;
// Lifetime of the underlying `DeviceKernelInfo` is tied to the availability of
// the `sycl_device_binaries` corresponding to this kernel. In other words, once
// user library is unloaded (see __sycl_unregister_lib), program manager
// destroys this `DeviceKernelInfo` object and the reference returned from here
// becomes stale.
_LIBSYCL_EXPORT DeviceKernelInfo &getDeviceKernelInfo(std::string_view);

template <class KernelName>
DeviceKernelInfo &getDeviceKernelInfo(std::string_view KernelNameStr) {
  static DeviceKernelInfo &Info = getDeviceKernelInfo(KernelNameStr);
  return Info;
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_DETAIL_GET_DEVICE_KERNEL_INFO_HPP
