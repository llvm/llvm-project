//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL queue class, which
/// schedules kernels on a device.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_QUEUE_HPP
#define _LIBSYCL___IMPL_QUEUE_HPP

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/context.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/handler.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/property_list.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
#include <sycl/__impl/detail/kernel_submission.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;

namespace detail {
class MockQueue;
class QueueImpl;

template <typename, typename T> struct CheckFunctionCallOperator {
  static_assert(std::integral_constant<T, false>::value,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct CheckFunctionCallOperator<F, RetT(Args...)> {
private:
  template <typename T>
  static constexpr auto check(T *) ->
      typename std::is_same<decltype(std::declval<std::add_const_t<T>>()
                                         .operator()(std::declval<Args>()...)),
                            RetT>::type;

  template <typename> static constexpr std::false_type check(...);

  using type = decltype(check<F>(0));

public:
  static constexpr bool value = type::value;
};

class TypelessCGF {
public:
  template <class T>
  TypelessCGF(T &&F)
      // NOTE: Even if `F` is a pointer to a function, `&F` is a pointer to a
      // pointer to a function and as such can be cast to `void *` (pointer to
      // a function cannot be cast).
      : MObject(static_cast<const void *>(&F)),
        MInvokerF(&Invoker<std::remove_reference_t<T>>::call) {}
  ~TypelessCGF() = default;

  TypelessCGF(const TypelessCGF &) = delete;
  TypelessCGF(TypelessCGF &&) = delete;
  TypelessCGF &operator=(const TypelessCGF &) = delete;
  TypelessCGF &operator=(TypelessCGF &&) = delete;

  void operator()(handler &CGH) const { MInvokerF(MObject, CGH); }

private:
  // SYCL 2020 command group function object is a type that is callable with
  // operator(), takes a reference to a command group handler, and defines
  // a command group that can be submitted by a queue. The function object can
  // be a named type, lambda expression or std::function.
  template <typename T> struct Invoker {
    static void call(const void *Object, handler &CGH) {
      (*const_cast<T *>(static_cast<const T *>(Object)))(CGH);
    }
  };
  const void *MObject;
  using InvokerTy = void (*)(const void *, handler &);
  const InvokerTy MInvokerF;
};

} // namespace detail

/// Schedules kernels and memory operations submitted to it on a single device.
///
/// A queue constructed without a context uses the default context of the
/// platform that contains its device.
///
/// \sa SYCL 2020 4.6.5. Queue class.
class _LIBSYCL_EXPORT queue : private detail::KernelSubmissionBase<queue> {
public:
  queue(const queue &rhs) = default;
  queue(queue &&rhs) = default;
  queue &operator=(const queue &rhs) = default;
  queue &operator=(queue &&rhs) = default;
  ~queue() = default;

  friend bool operator==(const queue &lhs, const queue &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const queue &lhs, const queue &rhs) {
    return !(lhs == rhs);
  }

  /// Constructs a SYCL queue using the device selected by default_selector_v.
  ///
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if no device is available.
  explicit queue(const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v), async_handler{},
              propList) {}

  /// \overload
  ///
  /// \param asyncHandler receives the asynchronous errors of this queue.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if no device is available.
  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v), asyncHandler,
              propList) {}

  /// \overload
  ///
  /// \param deviceSelector a callable that takes a device and returns an int
  /// score; the device with the highest score is chosen, and a negative score
  /// rejects a device.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if \p deviceSelector
  /// rejects all devices.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), async_handler{}, propList) {
  }

  /// \overload
  ///
  /// \param deviceSelector a callable that takes a device and returns an int
  /// score; the device with the highest score is chosen, and a negative score
  /// rejects a device.
  /// \param asyncHandler receives the asynchronous errors of this queue.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if \p deviceSelector
  /// rejects all devices.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), asyncHandler, propList) {}

  /// \overload
  ///
  /// \param syclDevice is the device the queue submits commands to.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const property_list &propList = {})
      : queue(syclDevice, async_handler{}, propList) {}

  /// \overload
  ///
  /// \param syclDevice is the device the queue submits commands to.
  /// \param asyncHandler receives the asynchronous errors of this queue.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(syclDevice.get_platform().khr_get_default_context(), syclDevice,
              asyncHandler, propList) {}

  /// \overload
  ///
  /// \param syclContext is the context to associate the queue with.
  /// \param deviceSelector a callable that takes a device and returns an int
  /// score; the device with the highest score is chosen, and a negative score
  /// rejects a device.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if \p deviceSelector
  /// rejects all devices.
  /// \throw sycl::exception with sycl::errc::invalid if \p syclContext does
  /// not contain the selected device.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(syclContext, detail::SelectDevice(deviceSelector), propList) {}

  /// \overload
  ///
  /// \param syclContext is the context to associate the queue with.
  /// \param deviceSelector a callable that takes a device and returns an int
  /// score; the device with the highest score is chosen, and a negative score
  /// rejects a device.
  /// \param asyncHandler receives the asynchronous errors of this queue.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::runtime if \p deviceSelector
  /// rejects all devices.
  /// \throw sycl::exception with sycl::errc::invalid if \p syclContext does
  /// not contain the selected device.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const context &syclContext,
                 const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(syclContext, detail::SelectDevice(deviceSelector), asyncHandler,
              propList) {}

  /// \overload
  ///
  /// \param syclContext is the context to associate the queue with.
  /// \param syclDevice is the device the queue submits commands to.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::invalid if \p syclContext does
  /// not contain \p syclDevice.
  explicit queue(const context &syclContext, const device &syclDevice,
                 const property_list &propList = {});

  /// \overload
  ///
  /// \param syclContext is the context to associate the queue with.
  /// \param syclDevice is the device the queue submits commands to.
  /// \param asyncHandler receives the asynchronous errors of this queue.
  /// \param propList is a list of properties for queue construction.
  /// \throw sycl::exception with sycl::errc::invalid if \p syclContext does
  /// not contain \p syclDevice.
  explicit queue(const context &syclContext, const device &syclDevice,
                 const async_handler &asyncHandler,
                 const property_list &propList = {});

  /// \return the SYCL backend associated with this queue.
  backend get_backend() const noexcept;

  /// \return the associated SYCL context.
  context get_context() const;

  /// \return the SYCL device this queue was constructed with.
  device get_device() const;

  /// Equivalent to has_property<property::queue::in_order>().
  ///
  /// \return true if and only if the queue is in order.
  bool is_in_order() const;

  /// \return the value of the queue information descriptor Param.
  template <typename Param> typename Param::return_type get_info() const;

  /// \return the value of the SYCL backend-specific information descriptor
  /// Param of this queue.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Blocks the calling thread until all commands previously submitted to this
  /// queue have completed. Synchronous errors are reported through SYCL
  /// exceptions.
  void wait();

  /// Blocks the calling thread until all commands previously submitted to this
  /// queue have completed. Synchronous errors are reported through SYCL
  /// exceptions. At least all unconsumed asynchronous errors held by this queue
  /// are passed to the async_handler associated with the queue.
  void wait_and_throw();

  /// Checks to see if any unconsumed asynchronous errors have been produced by
  /// the queue and if so reports them by passing them to the async_handler
  /// associated with the queue.
  void throw_asynchronous();

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName = detail::AutoName, typename KernelType>
  event single_task(const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>(std::vector<event>{},
                                               kernelFunc);
  }

  /// \overload
  ///
  /// \param depEvent is an event that the kernel depends on.
  /// \param kernelFunc is the kernel functor or lambda.
  template <typename KernelName = detail::AutoName, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>(std::vector<event>{depEvent},
                                               kernelFunc);
  }

  /// \overload
  ///
  /// \param depEvents is a collection of events that the kernel depends on.
  /// \param kernelFunc is the kernel functor or lambda.
  template <typename KernelName = detail::AutoName, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    static_assert(
        detail::CheckFunctionCallOperator<std::remove_reference_t<KernelType>,
                                          void()>::value,
        "Invalid kernel function signature.");

    setKernelLaunchParams(depEvents);
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    submitSingleTask<NameT, KernelType>(kernelFunc);
    return getLastEvent();
  }

  // TODO: Rest will represent reduction types once it is supported.

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent is an event that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent is an event that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent is an event that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvents is a collection of events that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvents is a collection of events that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvents is a collection of events that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param executionRange specifies the global and local work spaces of the
  /// kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, Rest &&...rest) {
    return parallel_for<KernelName, Dims, Rest...>(
        executionRange, std::vector<event>{}, std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param executionRange specifies the global and local work spaces of the
  /// kernel.
  /// \param depEvent is an event that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, event depEvent,
                     Rest &&...rest) {
    return parallel_for<KernelName, Dims, Rest...>(executionRange,
                                                   std::vector<event>{depEvent},
                                                   std::forward<Rest>(rest)...);
  }

  /// \overload
  ///
  /// \param executionRange specifies the global and local work spaces of the
  /// kernel.
  /// \param depEvents is a collection of events that the kernel depends on.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange,
                     const std::vector<event> &depEvents, Rest &&...rest) {
    detail::checkNDRangeAndThrow(executionRange);
    return parallelForImpl<KernelName>(executionRange, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// Submits a memory copy operation from one USM or host pointer to another;
  /// USM pointers must be accessible on the device associated with the queue.
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \return an event that represents the status of the operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes) {
    return memcpy(dest, src, numBytes, std::vector<event>{});
  }

  /// \overload
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \param depEvent is an event that represents a dependency for the
  /// operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               event depEvent) {
    return memcpy(dest, src, numBytes, std::vector<event>{depEvent});
  }

  /// \overload
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \param depEvents is a vector of events that represent dependencies for the
  /// operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               const std::vector<event> &depEvents);

  /// Submits a memset operation on a USM allocation that must be accessible
  /// on the device associated with the queue, equivalent to a fill operation
  /// with an unsigned char pattern.
  ///
  /// \param ptr is the pointer to memory to be set.
  /// \param value is the value the memory should be filled with, interpreted
  /// as an unsigned char.
  /// \param numBytes is the number of bytes to set.
  /// \return an event that represents the status of the operation.
  event memset(void *ptr, int value, std::size_t numBytes) {
    return memset(ptr, value, numBytes, std::vector<event>{});
  }

  /// \overload
  ///
  /// \param ptr is the pointer to memory to be set.
  /// \param value is the value the memory should be filled with, interpreted
  /// as an unsigned char.
  /// \param numBytes is the number of bytes to set.
  /// \param depEvent is an event that represents a dependency for the
  /// operation.
  event memset(void *ptr, int value, std::size_t numBytes, event depEvent) {
    return memset(ptr, value, numBytes, std::vector<event>{depEvent});
  }

  /// \overload
  ///
  /// \param ptr is the pointer to memory to be set.
  /// \param value is the value the memory should be filled with, interpreted
  /// as an unsigned char.
  /// \param numBytes is the number of bytes to set.
  /// \param depEvents is a vector of events that represent dependencies for the
  /// operation.
  event memset(void *ptr, int value, std::size_t numBytes,
               const std::vector<event> &depEvents) {
    return fill(ptr, static_cast<unsigned char>(value), numBytes, depEvents);
  }

  /// Submits a fill operation that replicates a pattern into a USM allocation
  /// that must be accessible on the device associated with the queue.
  ///
  /// \param ptr is the pointer to memory to be filled.
  /// \param pattern is the pattern to be replicated.
  /// \param count is the number of times the pattern is replicated.
  /// \return an event that represents the status of the operation.
  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count) {
    return fill(ptr, pattern, count, std::vector<event>{});
  }

  /// \overload
  ///
  /// \param ptr is the pointer to memory to be filled.
  /// \param pattern is the pattern to be replicated.
  /// \param count is the number of times the pattern is replicated.
  /// \param depEvent is an event that represents a dependency for the
  /// operation.
  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count, event depEvent) {
    return fill(ptr, pattern, count, std::vector<event>{depEvent});
  }

  /// \overload
  ///
  /// \param ptr is the pointer to memory to be filled.
  /// \param pattern is the pattern to be replicated.
  /// \param count is the number of times the pattern is replicated.
  /// \param depEvents is a vector of events that represent dependencies for the
  /// operation.
  template <typename T>
  event fill(void *ptr, const T &pattern, std::size_t count,
             const std::vector<event> &depEvents) {
    return fillImpl(ptr, &pattern, sizeof(T), count, depEvents);
  }

  /// Immediately calls the command group function object, which may submit no
  /// more than one command to this queue for execution on the associated
  /// device.
  ///
  /// \param cgf is the command group function object.
  /// \return an event that represents the status of the submitted command.
  template <typename T>
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event>
  submit(T cgf) {
    return submitWithHandler(cgf);
  }

  /// \overload
  ///
  /// \param cgf is the command group function object.
  /// \param secondaryQueue a queue on which this command group may be scheduled
  /// for execution on a kernel error. Unused (See SYCL 2020 3.9.10. Fallback
  /// mechanism).
  template <typename T>
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event>
  submit(T cgf, [[maybe_unused]] queue &secondaryQueue) {
    return submitWithHandler(cgf);
  }

  /// Provides hints to the runtime library that data can be made available
  /// on a device earlier than Unified Shared Memory would normally require it
  /// to be available.
  ///
  /// \param ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param numBytes is a number of bytes to be prefetched.
  /// \return an event representing prefetch operation.
  event prefetch(const void *ptr, std::size_t numBytes) {
    return prefetch(ptr, numBytes, std::vector<event>{});
  }

  /// \overload
  ///
  /// \param ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param numBytes is a number of bytes to be prefetched.
  /// \param depEvent is an event that represents a dependency for the
  /// operation.
  event prefetch(const void *ptr, std::size_t numBytes, event depEvent) {
    return prefetch(ptr, numBytes, std::vector<event>{depEvent});
  }

  /// \overload
  ///
  /// \param ptr is a USM pointer to the memory to be prefetched to the device.
  /// \param numBytes is a number of bytes to be prefetched.
  /// \param depEvents is a vector of events that represent dependencies for the
  /// operation.
  event prefetch(const void *ptr, std::size_t numBytes,
                 const std::vector<event> &depEvents);

private:
  template <typename KernelName, int Dims, template <int> class Range,
            typename... Rest>
  event parallelForImpl(Range<Dims> numWorkItems,
                        const std::vector<event> &depEvents, Rest &&...rest) {
    setKernelLaunchParams(depEvents, numWorkItems);

    detail::KernelSubmissionBase<queue>::template parallelForImpl<KernelName>(
        numWorkItems, std::forward<Rest>(rest)...);
    return getLastEvent();
  }

  /// Passes kernel dependencies and execution range to the runtime.
  /// \param Events a collection of events representing dependencies of the
  /// kernel to submit.
  /// \param Range a unified view of the kernel execution range.
  void setKernelLaunchParams(const std::vector<event> &Events,
                             const detail::UnifiedRangeView &Range = {});

  /// Passes kernel arguments to runtime.
  /// \param KernelInfo the information for the kernel being invoked.
  /// \param ArgData a pointer to the kernel argument.
  /// \param ArgSize the size of the kernel argument.
  void submitKernelImpl(detail::DeviceKernelInfo &KernelInfo, void *ArgData,
                        std::size_t ArgSize);

  /// \return an event representing the last kernel invocation.
  event getLastEvent();

  /// Implements fill() for a type-erased pattern.
  event fillImpl(void *Ptr, const void *Pattern, std::size_t PatternSize,
                 std::size_t Count, const std::vector<event> &DepEvents);

  event submitWithHandler(const detail::TypelessCGF &CGF);

  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  friend sycl::detail::ImplUtils;
  friend sycl::detail::MockQueue;
  friend class detail::KernelSubmissionBase<queue>;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
