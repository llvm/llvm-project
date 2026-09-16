//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the class that aggregates information
/// specific to device kernels (i.e. information that is uniform between
/// different submissions of the same kernel).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DEVICE_KERNEL_INFO
#define _LIBSYCL_DEVICE_KERNEL_INFO

#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <string_view>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class DeviceImageManager;

// TODO: Pointers to instances of this class are supported to be stored in
// header function templates as a static variable to avoid repeated runtime
// lookup overhead.
class DeviceKernelInfo {
public:
  /// Constructs a device kernel info instance.
  ///
  /// \param KernelName the name of the kernel.
  /// \param DeviceImage the device image containing device code of this kernel.
  DeviceKernelInfo(std::string_view KernelName, DeviceImageManager &DeviceImage)
      : MName(KernelName), MDeviceImage(DeviceImage) {}

  /// \return the name of this kernel.
  std::string_view getName() { return MName; }

  /// \return the device image containing the device code of this kernel.
  DeviceImageManager &getDeviceImage() const { return MDeviceImage; }

private:
  std::string_view MName;
  DeviceImageManager &MDeviceImage;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_KERNEL_INFO
