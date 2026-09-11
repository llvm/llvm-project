//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

ContextImpl::ContextImpl(std::vector<DeviceImpl *> &&DeviceList,
                         const async_handler &AsyncHandler,
                         const property_list &PropList, Private)
    : MAsyncHandler(AsyncHandler), MDevices(std::move(DeviceList)) {
  // TODO: Remove this when property_list is implemented
  std::ignore = PropList;

  std::vector<ol_device_handle_t> DeviceIds;
  DeviceIds.reserve(MDevices.size());
  for (DeviceImpl *D : MDevices) {
    assert(D && "Device list must not contain null entries");
    DeviceIds.push_back(D->getOLHandle());
  }

  auto Result = callNoCheck(olCreateContext, DeviceIds.size(), DeviceIds.data(),
                            &MOffloadContext);
  if (isFailed(Result)) {
    if (Result->Code == OL_ERRC_INVALID_SIZE)
      throw sycl::exception(make_error_code(errc::invalid),
                            "Device list must not be empty");

    checkAndThrow(Result);
  }
}

ContextImpl::~ContextImpl() {
  assert(MOffloadContext && "Context must be created in ctor");
  // liboffload does not reference-count contexts: every resource tied to a
  // context must be released before olDestroyContext, otherwise it is left in
  // an undefined state. MPrograms is a member, so it would be destroyed only
  // after this destructor body has run.
  releaseAllPrograms();
  std::ignore = olDestroyContext(MOffloadContext);
}

PlatformImpl &ContextImpl::getPlatformImpl() const {
  return MDevices[0]->getPlatformImpl();
}

void ContextImpl::iterateDevices(
    const std::function<void(DeviceImpl *)> &callback) const {
  for (DeviceImpl *Device : MDevices)
    callback(Device);
}

backend ContextImpl::getBackend() const { return MDevices[0]->getBackend(); }

ol_symbol_handle_t
ContextImpl::getOrCreateKernel(const DeviceImageManager &DeviceImage,
                               ol_device_handle_t DeviceHandle,
                               std::string_view KernelName) {
  std::lock_guard<std::mutex> Guard(MProgramCacheMutex);

  auto ImageIt = MPrograms.try_emplace(&DeviceImage).first;
  ProgramsByDeviceT &ProgramsForImage = ImageIt->second;

  auto ProgramIt = ProgramsForImage.find(DeviceHandle);
  if (ProgramIt == ProgramsForImage.end()) {
    // Constructing a ProgramWrapper calls olCreateProgram, so try_emplace is
    // used rather than emplace: the latter would build a program even when one
    // is already cached, only to destroy it again.
    try {
      ProgramIt = ProgramsForImage
                      .try_emplace(DeviceHandle, MOffloadContext, DeviceHandle,
                                   DeviceImage)
                      .first;
    } catch (...) {
      // Do not leave an empty entry behind if program creation failed.
      if (ProgramsForImage.empty())
        MPrograms.erase(ImageIt);
      throw;
    }
  }

  return ProgramIt->second.getOrCreateKernel(KernelName);
}

void ContextImpl::releaseProgramsForImage(
    const DeviceImageManager &DeviceImage) {
  std::lock_guard<std::mutex> Guard(MProgramCacheMutex);
  MPrograms.erase(&DeviceImage);
}

void ContextImpl::releaseAllPrograms() {
  std::lock_guard<std::mutex> Guard(MProgramCacheMutex);
  MPrograms.clear();
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
