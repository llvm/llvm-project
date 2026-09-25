//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the EventImpl class, which
/// implements sycl::event functionality.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_EVENT_IMPL
#define _LIBSYCL_EVENT_IMPL

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>

#include <OffloadAPI.h>

#include <memory>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class PlatformImpl;

class EventImpl {
  // Helper to limit EventImpl creation.
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  /// Constructs a SYCL event instance using the provided
  /// offload event instance.
  ///
  /// \param Event is the raw offload library handle representing the event.
  /// \param Platform is the platform this event belongs to.
  EventImpl(ol_event_handle_t Event, PlatformImpl &Platform, PrivateTag)
      : MOffloadEvent(Event), MPlatform(Platform) {}

  /// Constructs a default, immediately ready event.
  /// The event is constructed as though it were created from a
  /// default-constructed queue. Therefore, its backend is the same as the
  /// backend of the device selected by default_selector_v.
  /// \throw sycl::exception with errc::runtime if no default device is
  /// available.
  EventImpl(PrivateTag);

  static std::shared_ptr<EventImpl>
  createEventWithHandle(ol_event_handle_t Event, PlatformImpl &Platform,
                        std::vector<std::shared_ptr<EventImpl>> &&WaitList) {
    auto E = std::make_shared<EventImpl>(Event, Platform, PrivateTag{});
    E->MWaitList = std::move(WaitList);
    return E;
  }

  static std::shared_ptr<EventImpl> createDefaultEvent() {
    return std::make_shared<EventImpl>(PrivateTag{});
  }

  /// Releases the handle to the corresponding liboffload event.
  ~EventImpl();

  /// \return the sycl::backend associated with this event.
  backend getBackend() const noexcept;

  /// Waits for completion of the corresponding command and its dependencies.
  void wait();

  /// \return the liboffload handle that this SYCL event represents.
  ol_event_handle_t getHandle() { return MOffloadEvent; }

  /// \return the platform implementation object this event belongs to.
  const PlatformImpl &getPlatformImpl() const { return MPlatform; }

  /// Blocks until all commands associated with this event and any dependency
  /// events have completed. Passes at least all unconsumed asynchronous errors
  /// held by the queues (or their associated contexts) that were used to
  /// enqueue commands associated with this event and any dependency events, to
  /// the appropriate async_handler.
  void waitAndThrow();

  /// \return the list of event implementation objects that this event waits for
  /// in the dependence graph.
  const std::vector<std::shared_ptr<EventImpl>> &getWaitList() const {
    return MWaitList;
  }

private:
  ol_event_handle_t MOffloadEvent{};
  PlatformImpl &MPlatform;

  std::vector<std::shared_ptr<EventImpl>> MWaitList;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_EVENT_IMPL
