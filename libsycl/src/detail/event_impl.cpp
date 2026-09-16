//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/device_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/global_objects.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

EventImpl::EventImpl(PrivateTag)
    : MPlatform(detail::getSyclObjImpl(device(default_selector_v))
                    ->getPlatformImpl()) {}

EventImpl::~EventImpl() {
  if (MOffloadEvent)
    std::ignore = olDestroyEvent(MOffloadEvent);
}

backend EventImpl::getBackend() const noexcept {
  // TODO: to handle default constructed.
  return MPlatform.getBackend();
}

void EventImpl::wait() {
  // MOffloadEvent == nullptr when the event is default constructed. Default
  // constructed event is immediately completed.
  if (!MOffloadEvent)
    return;

  callAndThrow(olSyncEvent, MOffloadEvent);
}

void EventImpl::waitAndThrow() {
  wait();
  flushAsyncExceptions();
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
