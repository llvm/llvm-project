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
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/handler.hpp>
#include <sycl/__impl/property_list.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
#include <sycl/__impl/detail/kernel_submission.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>
#include <sycl/__impl/exception.hpp>

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
} // namespace detail

class TypelessCGF {
  // SYCL 2020 command group function object is a type which is callable with
  // operator() that takes a reference to a command group handler, that defines
  // a command group which can be submitted by a queue. The function object can
  // be a named type, lambda expression or std::function.
  template <typename T> struct Invoker {
    static void call(const void *Object, handler &CGH) {
      (*const_cast<T *>(static_cast<const T *>(Object)))(CGH);
    }
  };
  const void *Object;
  using InvokerTy = void (*)(const void *, handler &);
  const InvokerTy InvokerF;

public:
  template <class T>
  TypelessCGF(T &&F)
      // NOTE: Even if `F` is a pointer to a function, `&F` is a pointer to a
      // pointer to a function and as such can be casted to `void *` (pointer to
      // a function cannot be casted).
      : Object(static_cast<const void *>(&F)),
        InvokerF(&Invoker<std::remove_reference_t<T>>::call) {}
  ~TypelessCGF() = default;

  TypelessCGF(const TypelessCGF &) = delete;
  TypelessCGF(TypelessCGF &&) = delete;
  TypelessCGF &operator=(const TypelessCGF &) = delete;
  TypelessCGF &operator=(TypelessCGF &&) = delete;

  void operator()(handler &CGH) const { InvokerF(Object, CGH); }
};

// SYCL 2020 4.6.5. Queue class.
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

  /// Constructs a SYCL queue instance using the device returned by an instance
  /// of default_selector.
  ///
  /// \param propList is a list of properties for queue construction.
  explicit queue(const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v),
              detail::defaultAsyncHandler, propList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// returned by an instance of default_selector.
  ///
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(default_selector_v), asyncHandler,
              propList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param deviceSelector is a SYCL 2020 Device Selector, a simple callable
  /// that takes a device and returns an int
  /// \param propList is a list of properties for queue construction.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), detail::defaultAsyncHandler,
              propList) {}

  /// Constructs a SYCL queue instance using the device identified by the
  /// device selector provided.
  /// \param deviceSelector is a SYCL 2020 Device Selector, a simple callable
  /// that takes a device and returns an int
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  template <
      typename DeviceSelector,
      typename = detail::EnableIfDeviceSelectorIsInvocable<DeviceSelector>>
  explicit queue(const DeviceSelector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue(detail::SelectDevice(deviceSelector), asyncHandler, propList) {}

  /// Constructs a SYCL queue instance using the device provided.
  ///
  /// \param syclDevice is an instance of SYCL device.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const property_list &propList = {})
      : queue(syclDevice, detail::defaultAsyncHandler, propList) {}

  /// Constructs a SYCL queue instance with an async_handler using the device
  /// provided.
  ///
  /// \param syclDevice is an instance of SYCL device.
  /// \param asyncHandler is a SYCL asynchronous exception handler.
  /// \param propList is a list of properties for queue construction.
  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
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

  /// Queries the queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries the queue for SYCL backend-specific information.
  ///
  /// The return type depends on the information being queried.
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

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param depEvent is an event that specifies the kernel dependency.
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName = detail::AutoName, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>(std::vector<event>{depEvent},
                                               kernelFunc);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param depEvents is a collection of events that specify the kernel
  /// dependencies.
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName = detail::AutoName, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    static_assert(
        detail::CheckFunctionCallOperator<std::remove_reference_t<KernelType>,
                                          void()>::value,
        "Invalid kernel function signature.");

    setKernelDependencies(depEvents);
    setKernelRange({});
    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    submitSingleTask<NameT, KernelType>(kernelFunc);
    return getLastEvent();
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems, std::vector<event>{},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent adds a requirement that the action represented by depEvent
  /// must complete before executing this kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent adds a requirement that the action represented by depEvent
  /// must complete before executing this kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel.
  /// \param depEvent adds a requirement that the action represented by depEvent
  /// must complete before executing this kernel.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, event depEvent, Rest &&...rest) {
    return parallel_for<KernelName, Rest...>(numWorkItems,
                                             std::vector<event>{depEvent},
                                             std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel
  /// \param depEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<1> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel
  /// \param depEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<2> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type, for the specified range.
  ///
  /// \param numWorkItems specifies the global work space of the kernel
  /// \param depEvents is a vector of events that specifies the kernel
  /// dependencies.
  /// \param rest acts as if it was "const KernelType &KernelFunc".
  // TODO: Rest will represent reduction types once it is supported.
  template <typename KernelName = detail::AutoName, typename... Rest>
  event parallel_for(range<3> numWorkItems, const std::vector<event> &depEvents,
                     Rest &&...rest) {
    return parallelForImpl<KernelName>(numWorkItems, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, Rest &&...rest) {
    return parallel_for<KernelName, Dims, Rest...>(
        executionRange, std::vector<event>{}, std::forward<Rest>(rest)...);
  }

  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange, event depEvent,
                     Rest &&...rest) {
    return parallel_for<KernelName, Dims, Rest...>(executionRange,
                                                   std::vector<event>{depEvent},
                                                   std::forward<Rest>(rest)...);
  }

  template <typename KernelName = detail::AutoName, int Dims, typename... Rest>
  event parallel_for(nd_range<Dims> executionRange,
                     const std::vector<event> &depEvents, Rest &&...rest) {
    detail::checkNDRangeAndThrow(executionRange);
    return parallelForImpl<KernelName>(executionRange, depEvents,
                                       std::forward<Rest>(rest)...);
  }

  /// Submits a memory copy operation from one USM or host pointer to another.
  /// USM pointers must be accessible on the device associated with the queue.
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \return an event that represents the status of the operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes) {
    return memcpy(dest, src, numBytes, std::vector<event>{});
  }

  /// Submits a memory copy operation from one USM or host pointer to another.
  /// USM pointers must be accessible on the device associated with the queue.
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \param depEvent is an event that represents a dependency for the
  /// operation.
  /// \return an event that represents the status of the operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               event depEvent) {
    return memcpy(dest, src, numBytes, std::vector<event>{depEvent});
  }

  /// Submits a memory copy operation from one USM or host pointer to another.
  /// USM pointers must be accessible on the device associated with the queue.
  ///
  /// \param dest is the pointer to copy to.
  /// \param src is the pointer to copy from.
  /// \param numBytes is the number of bytes to copy.
  /// \param depEvents is a vector of events that represent dependencies for the
  /// operation.
  /// \return an event that represents the status of the operation.
  event memcpy(void *dest, const void *src, std::size_t numBytes,
               const std::vector<event> &depEvents);

  /// Immediately calls the command group function object.
  ///
  /// The command group may submit no more than one command to this queue for
  /// execution on the associated device.
  ///
  /// \param cgf command group function object.
  /// \return an event that represents the status of the submitted command.
  template <typename T>
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event>
  submit(T cgf) {
    return submitWithHandler(cgf);
  }

  /// Immediately calls the command group function object.
  ///
  /// The command group may submit no more than one command to this queue for
  /// execution on the associated device. On a kernel error, this command group
  /// may be scheduled for execution on \p secondaryQueue.
  ///
  /// \param cgf command group function object.
  /// \param secondaryQueue queue used as a fallback for kernel errors. Unused
  /// (See SYCL 2020 3.9.10. Fallback mechanism).
  /// \return an event that represents the status of the submitted command.
  template <typename T>
  std::enable_if_t<std::is_invocable_r_v<void, T, handler &>, event>
  submit(T cgf, [[maybe_unused]] queue &secondaryQueue) {
    return submitWithHandler(cgf);
  }

private:
  template <typename KernelName, int Dims, template <int> class Range,
            typename... Rest>
  event parallelForImpl(Range<Dims> numWorkItems,
                        const std::vector<event> &depEvents, Rest &&...rest) {
    setKernelDependencies(depEvents);
    setKernelRange(numWorkItems);

    detail::KernelSubmissionBase<queue>::template parallelForImpl<KernelName>(
        numWorkItems, std::forward<Rest>(rest)...);
    return getLastEvent();
  }

  template <typename KN, typename ArgT>
  void submitKernelFromLaunch(const char *KernelName, ArgT &FirstArg) {
    submitKernelImpl(detail::getDeviceKernelInfo<KN>(KernelName), &FirstArg,
                     sizeof(FirstArg));
  }

  /// Passes kernel dependency events to the runtime.
  /// \param Events a collection of events representing dependencies of the
  /// kernel to submit.
  void setKernelDependencies(const std::vector<event> &Events);

  /// Passes kernel execution range to the runtime.
  /// \param Range a unified view of the kernel execution range.
  void setKernelRange(const detail::UnifiedRangeView &Range);

  /// Passes kernel arguments to runtime.
  /// \param KernelInfo the information for the kernel being invoked.
  /// \param ArgData a pointer to the kernel argument.
  /// \param ArgSize the size of the kernel argument.
  void submitKernelImpl(detail::DeviceKernelInfo &KernelInfo, void *ArgData,
                        size_t ArgSize);

  /// \return an event representing last kernel invocation.
  event getLastEvent();

  event submitWithHandler(const TypelessCGF &CGF);

  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  friend sycl::detail::ImplUtils;
  friend sycl::detail::MockQueue;
  friend class sycl::handler;
  friend class detail::KernelSubmissionBase<queue>;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
