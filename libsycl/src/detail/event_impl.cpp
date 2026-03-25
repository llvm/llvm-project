//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/event_impl.hpp>
#include <detail/platform_impl.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {

EventImpl::~EventImpl() {
  if (MOffloadEvent)
    std::ignore = olDestroyEvent(MOffloadEvent);
}

backend EventImpl::getBackend() const noexcept {
  // TODO: to handle default cosntructed
  //  The event is constructed as though it were created from a
  //  default-constructed queue. Therefore, its backend is the same as the
  //  backend of the device selected by default_selector_v.
  return MPlatform.getBackend();
}

void EventImpl::wait() {
  // MOffloadEvent == nullptr when event is default constructed. Default
  // constructed event is immediately  ready.
  if (!MOffloadEvent)
    return;

  callAndThrow(olSyncEvent, MOffloadEvent);
}

} // namespace detail
_LIBSYCL_END_NAMESPACE_SYCL
