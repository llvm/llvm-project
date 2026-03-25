//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_EVENT_IMPL
#define _LIBSYCL_EVENT_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class PlatformImpl;

class EventImpl {
  // Helper to limit EventImpl creation.
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  /// Constructs a SYCL event  instance using the provided
  /// offload event instance.
  ///
  /// \param Event is a raw offload library handle representing event.
  /// \param Platform is a platform this event belongs to.
  EventImpl(ol_event_handle_t Event, PlatformImpl &Platform, PrivateTag)
      : MOffloadEvent(Event), MPlatform(Platform) {}

  static std::shared_ptr<EventImpl>
  createEventWithHandle(ol_event_handle_t Event, PlatformImpl &Queue) {
    return std::make_shared<EventImpl>(Event, Queue, PrivateTag{});
  }

  /// Releases handle to the corresponding liboffload event.
  ~EventImpl();

  /// \return the sycl::backend associated with this event.
  backend getBackend() const noexcept;

  /// Waits for completion of the corresponding kernel and its dependencies.
  void wait();

  /// \return liboffload handle that this SYCL event represents.
  ol_event_handle_t getHandle() { return MOffloadEvent; }

  /// \return a platform implementation object this event belongs to.
  const PlatformImpl &getPlatformImpl() const { return MPlatform; }

private:
  ol_event_handle_t MOffloadEvent{};
  PlatformImpl &MPlatform;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_EVENT_IMPL
