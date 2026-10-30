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

#ifndef _LIBSYCL_SRC_DETAIL_QUEUE_IMPL_HPP
#define _LIBSYCL_SRC_DETAIL_QUEUE_IMPL_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/queue.hpp>

#include <OffloadAPI.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

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

  /// Constructs a SYCL queue from a device using an AsyncHandler and
  /// a PropList.
  ///
  /// \param Context is a SYCL context the queue is associated with.
  /// \param Device is a SYCL device that is used to dispatch tasks
  /// submitted to the queue.
  /// \param AsyncHandler receives the asynchronous errors of this queue.
  /// \param PropList is a list of properties to use for queue construction.
  explicit QueueImpl(const std::shared_ptr<ContextImpl> &Context,
                     DeviceImpl &Device, const async_handler &AsyncHandler,
                     const property_list &PropList, PrivateTag);

  /// Constructs a QueueImpl with the provided arguments. Variadic helper.
  /// Restricts QueueImpl creation to std::shared_ptr allocations.
  template <typename... Ts>
  static std::shared_ptr<QueueImpl> create(Ts &&...Args) {
    return std::make_shared<QueueImpl>(std::forward<Ts>(Args)..., PrivateTag{});
  }

  /// \return the SYCL backend this queue is associated with.
  backend getBackend() const noexcept;

  /// \return the context implementation object this queue is associated with.
  ContextImpl &getContext() { return *MContext; }

  /// \return a weak pointer to the context implementation object this queue is
  /// associated with.
  std::weak_ptr<ContextImpl> getContextWeakPtr() const { return MContext; }

  /// \return the device implementation object this queue is associated with.
  DeviceImpl &getDevice() { return MDevice; }

  /// \return true if and only if the queue is in order.
  bool isInOrder() const { return MIsInOrder; }

  /// Waits for completion of all commands submitted to this queue.
  void wait();

  /// Waits for completion of all commands submitted to this queue and flushes
  /// unconsumed async errors to the appropriate async handlers.
  void waitAndThrow();

  /// Flushes unconsumed async errors to the appropriate async handlers.
  void throwAsynchronous();

  /// Enqueues a kernel to liboffload.
  /// Kernel dependencies and range must be passed in advance by calling
  /// setKernelLaunchParams.
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

  /// Sets event dependencies and execution range for the next kernel
  /// submission.
  ///
  /// \param Events a collection of events that the kernel depends on.
  /// \param Range a unified range view of the execution range.
  void setKernelLaunchParams(std::vector<EventImplPtr> &&Events,
                             const detail::UnifiedRangeView &Range);

  /// \overload
  ///
  /// \param Events a collection of events that the kernel depends on.
  /// \param Range a pre-converted liboffload kernel launch size args struct.
  void setKernelLaunchParams(std::vector<EventImplPtr> &&Events,
                             const ol_kernel_launch_size_args_t &Range);

  /// \return the async_handler associated with this queue, empty if the queue
  /// was constructed without one and its context has none either.
  const async_handler &getAsyncHandler() const { return MAsyncHandler; }

  /// Implements sycl::queue::memcpy.
  EventImplPtr memcpy(void *Dest, const void *Src, std::size_t NumBytes,
                      const std::vector<EventImplPtr> &DepEvents);

  /// Implements sycl::queue::fill for a type-erased pattern.
  EventImplPtr fill(void *Ptr, const void *Pattern, std::size_t PatternSize,
                    std::size_t Count,
                    const std::vector<EventImplPtr> &DepEvents);

  /// Submits a command group function to this queue.
  ///
  /// \param CGF is the command group function to invoke.
  /// \return an event impl object representing the submitted operation.
  EventImplPtr submitWithHandler(const TypelessCGF &CGF);

  /// Submits a dependency-only wait operation to this queue.
  ///
  /// \param DepEvents are the events that must complete before the wait
  /// operation completes.
  /// \return an event impl object representing the wait operation.
  EventImplPtr submitWait(const std::vector<EventImplPtr> &DepEvents);

  /// Implements sycl::queue::prefetch.
  EventImplPtr prefetch(const void *Ptr, std::size_t NumBytes,
                        const std::vector<EventImplPtr> &DepEvents);

private:
  void handleEventDependencies(const std::vector<EventImplPtr> &Deps);
  EventImplPtr createEvent(std::vector<EventImplPtr> &&Deps = {});

  // Queue features.
  ol_queue_handle_t MOffloadQueue = {};
  const bool MIsInOrder;
  const async_handler MAsyncHandler;
  const property_list MPropList;
  DeviceImpl &MDevice;
  const std::shared_ptr<ContextImpl> MContext;

  // Submit data.
  struct KernelSubmitInfo {
    KernelSubmitInfo() {}

    EventImplPtr LastEvent;
    ol_kernel_launch_size_args_t Range;
    std::vector<EventImplPtr> DepEvents;
  };
  inline static thread_local KernelSubmitInfo MCurrentSubmitInfo = {};
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SRC_DETAIL_QUEUE_IMPL_HPP
