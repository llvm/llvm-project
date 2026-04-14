//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/queue_impl.hpp>

#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

QueueImpl::QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag)
    : MIsInorder(false), MAsyncHandler(asyncHandler), MPropList(propList),
      MDevice(deviceImpl),
      MContext(MDevice.getPlatformImpl().getDefaultContext()) {
  callAndThrow(olCreateQueue, MDevice.getOLHandle(), &MOffloadQueue);
}

QueueImpl::~QueueImpl() {
  // TODO: consider where to report errors
  if (MOffloadQueue)
    std::ignore = olDestroyQueue(MOffloadQueue);
}

backend QueueImpl::getBackend() const noexcept { return MDevice.getBackend(); }

void QueueImpl::wait() { callAndThrow(olSyncQueue, MOffloadQueue); }

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
