//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the QueueImpl class, which implements
/// sycl::queue functionality.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_QUEUE_IMPL
#define _LIBSYCL_QUEUE_IMPL

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <OffloadAPI.h>

#include <memory>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class ContextImpl;
class DeviceImpl;
class EventImpl;

using EventImplPtr = std::shared_ptr<EventImpl>;

class QueueImpl : public std::enable_shared_from_this<QueueImpl> {
  struct PrivateTag {
    explicit PrivateTag() = default;
  };

public:
  ~QueueImpl();

  /// Constructs a SYCL queue from a device using an asyncHandler and
  /// a propList.
  ///
  /// \param deviceImpl is a SYCL device that is used to dispatch tasks
  /// submitted to the queue.
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties to use for queue construction.
  explicit QueueImpl(DeviceImpl &deviceImpl, const async_handler &asyncHandler,
                     const property_list &propList, PrivateTag);

  /// Constructs a QueueImpl with the provided arguments. Variadic helper.
  /// Restricts QueueImpl creation to std::shared_ptr allocations.
  template <typename... Ts>
  static std::shared_ptr<QueueImpl> create(Ts &&...args) {
    return std::make_shared<QueueImpl>(std::forward<Ts>(args)..., PrivateTag{});
  }

  /// \return the SYCL backend this queue is associated with.
  backend getBackend() const noexcept;

  /// \return the context implementation object this queue is associated with.
  ContextImpl &getContext() { return MContext; }

  /// \return the device implementation object this queue is associated with.
  DeviceImpl &getDevice() { return MDevice; }

  /// \return true if and only if the queue is in order.
  bool isInOrder() const { return MIsInorder; }

  /// Waits for completion of all commands submitted to this queue.
  void wait();

  /// Enqueues a kernel to liboffload.
  /// Kernel parameters like dependencies and range must be passed in advance by
  /// calling setKernelParameters.
  /// \param KernelInfo a kernel info that is uniform between different
  /// submissions of the same kernel.
  /// \param ArgData a pointer to kernel argument.
  /// \param ArgSize a size of kernel argument in bytes.
  void submitKernelImpl(DeviceKernelInfo &KernelInfo, void *ArgData,
                        size_t ArgSize);

  /// \return an event impl object that corresponds to the last kernel
  /// submission in the calling thread.
  EventImplPtr getLastEvent() {
    assert(MCurrentSubmitInfo.LastEvent &&
           "getLastEvent must be called after enqueue");
    return MCurrentSubmitInfo.LastEvent;
  }

  /// Sets kernel parameters to be used in the next submitKernelImpl call.
  /// Must be called prior to a submitKernelImpl call.
  /// \param Events a collection of events that the kernel depends on.
  /// \param Range a unified range view of the execution range.
  void setKernelParameters(std::vector<EventImplPtr> &&Events,
                           const detail::UnifiedRangeView &Range);

private:
  // Queue features.
  ol_queue_handle_t MOffloadQueue = {};
  const bool MIsInorder;
  const async_handler MAsyncHandler;
  const property_list MPropList;
  DeviceImpl &MDevice;
  ContextImpl &MContext;

  // Submit data.
  struct KernelSubmitInfo {
    EventImplPtr LastEvent;
    ol_kernel_launch_size_args_t Range;
    // TODO: consider storing EventImplPtr here, it will work with plain handle
    // only because submission is done within queue::submit call. Otherwise we
    // need to ensure that event handle is still alive by keeping our own copy
    // of EventImpl.
    std::vector<ol_event_handle_t> DepEvents;
  };
  inline static thread_local KernelSubmitInfo MCurrentSubmitInfo = {};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_QUEUE_IMPL
