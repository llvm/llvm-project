//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file contains the declaration of the SYCL handler class, which provides
/// the interface for the commands that can be executed inside the command group
/// scope.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_HANDLER_HPP
#define _LIBSYCL___IMPL_HANDLER_HPP

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
#include <sycl/__impl/detail/kernel_submission.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/index_space_classes.hpp>

#include <array>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class HandlerImpl;
class QueueImpl;
} // namespace detail

class _LIBSYCL_EXPORT handler : private detail::KernelSubmissionBase<handler> {
public:
  handler(const handler &) = delete;
  handler(handler &&) = delete;
  handler &operator=(const handler &) = delete;
  handler &operator=(handler &&) = delete;

  ~handler() = default;

  /// \copybrief handler::depends_on(std::vector<event>const&)
  ///
  /// \param depEvent is the event that must complete before this command
  /// group is executed.
  void depends_on(event depEvent) {
    return depends_on(std::vector<event>{depEvent});
  }

  /// \brief Adds event dependencies to this command group.
  ///
  /// \param depEvents are the events that must complete before this command
  /// group is executed.
  void depends_on(const std::vector<event> &depEvents) {
    MDepEvents.insert(MDepEvents.end(), depEvents.begin(), depEvents.end());
  }

  /// Defines a single-task kernel for this command group.
  ///
  /// \param kernelFunc is the kernel functor or lambda to execute.
  template <typename KernelName = detail::AutoName, typename KernelType>
  void single_task(const KernelType &kernelFunc) {
    setKernelRange();
    submitSingleTask<KernelName, KernelType>(kernelFunc);
  }

  /// Defines a one-dimensional range kernel for this command group.
  ///
  /// \param numWorkItems is the number of work-items in the kernel range.
  /// \param rest contains the kernel functor or lambda and its optional
  /// submission arguments.
  template <typename KernelName = detail::AutoName, typename... Rest>
  void parallel_for(range<1> numWorkItems, Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines a two-dimensional range kernel for this command group.
  ///
  /// \param numWorkItems is the number of work-items in the kernel range.
  /// \param rest contains the kernel functor or lambda and its optional
  /// submission arguments.
  template <typename KernelName = detail::AutoName, typename... Rest>
  void parallel_for(range<2> numWorkItems, Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines a three-dimensional range kernel for this command group.
  ///
  /// \param numWorkItems is the number of work-items in the kernel range.
  /// \param rest contains the kernel functor or lambda and its optional
  /// submission arguments.
  template <typename KernelName = detail::AutoName, typename... Rest>
  void parallel_for(range<3> numWorkItems, Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines an nd-range kernel for this command group.
  ///
  /// \param executionRange is the global and local range of the kernel.
  /// \param rest contains the kernel functor or lambda and its optional
  /// submission arguments.
  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  void parallel_for(nd_range<Dims> executionRange, Rest &&...rest) {
    detail::checkNDRangeAndThrow(executionRange);

    return parallelForImpl<KernelName>(executionRange,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines a memory copy operation for this command group.
  ///
  /// \param dest is the destination memory address.
  /// \param src is the source memory address.
  /// \param numBytes is the number of bytes to copy.
  void memcpy(void *dest, const void *src, std::size_t numBytes);

private:
  template <typename KernelName, int Dims, template <int> class Range,
            typename... Rest>
  void parallelForImpl(Range<Dims> numWorkItems, Rest &&...rest) {
    setKernelRange(numWorkItems);

    detail::KernelSubmissionBase<handler>::template parallelForImpl<KernelName>(
        numWorkItems, std::forward<Rest>(rest)...);
  }

  std::shared_ptr<detail::EventImpl> finalize();

  void submitKernelImpl(detail::DeviceKernelInfo &KernelInfo, void *ArgData,
                        size_t ArgSize);

  void setKernelRange(const detail::UnifiedRangeView &Range = {});

  handler(detail::HandlerImpl &HandlerImplVal) : MImpl(HandlerImplVal) {}

  std::vector<event> MDepEvents;

  detail::HandlerImpl &MImpl;

  friend sycl::detail::ImplUtils;
  friend class detail::QueueImpl;
  friend class detail::KernelSubmissionBase<handler>;
};

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_HANDLER_HPP
