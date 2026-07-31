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
    : MAsyncHandler(AsyncHandler), MDevices(DeviceList) {
  (void)PropList;

  assert(!MDevices.empty() && "Device list must not be empty");
  const PlatformImpl &RefPlatform = MDevices[0]->getPlatformImpl();

  std::vector<ol_device_handle_t> DeviceIds;
  DeviceIds.reserve(MDevices.size());
  for (DeviceImpl *D : MDevices) {
    assert(D && "Device list must not contain null entries");
    if (D->getPlatformImpl().getOLHandleRef() != RefPlatform.getOLHandleRef())
      throw exception(
          make_error_code(errc::invalid),
          "Can't add devices across platforms to a single context.");
    DeviceIds.push_back(D->getOLHandle());
  }

  callAndThrow(olCreateContext, DeviceIds.size(), DeviceIds.data(),
               &MOffloadContext);
}

ContextImpl::~ContextImpl() {
  assert(MOffloadContext && "Context must be created in ctor");
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

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
