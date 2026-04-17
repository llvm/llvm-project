//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_image_wrapper.hpp>

#include <detail/offload/offload_utils.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

ProgramWrapper::ProgramWrapper(ol_device_handle_t Device,
                               DeviceImageManager &DevImage) {
  assert(Device);

  callAndThrow(olCreateProgram, Device, DevImage.getRawData().ImageStart,
               DevImage.getSize(), &MProgram);
}

ProgramWrapper::~ProgramWrapper() {
  assert(MProgram);
  std::ignore = olDestroyProgram(MProgram);
  // TODO: define a way to report errors from dtors.
}

ol_program_handle_t
DeviceImageManager::getOrCreateProgram(ol_device_handle_t DeviceHandle) {
  const auto &[Iterator, Flag] = MPrograms.emplace(
      std::piecewise_construct, std::forward_as_tuple(DeviceHandle),
      std::forward_as_tuple(DeviceHandle, *this));
  return Iterator->second.getOLHandle();
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
