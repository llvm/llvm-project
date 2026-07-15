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
#include <sycl/__impl/property_list.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/kernel_arg_helpers.hpp>
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

// SYCL 2020 4.6.5. Queue class.
class _LIBSYCL_EXPORT queue {
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

    setKernelParameters(depEvents);
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

private:
  template <typename KernelName, int Dims, typename... Rest>
  event parallelForImpl(range<Dims> numWorkItems,
                        const std::vector<event> &depEvents, Rest &&...rest) {
    if constexpr (sizeof...(Rest) != 1)
      throw sycl::exception(errc::feature_not_supported,
                            "Reductions are not supported");
    setKernelParameters(depEvents, numWorkItems);

    using KernelType =
        std::decay_t<detail::nth_type_t<sizeof...(Rest) - 1, Rest...>>;
    using LambdaArgType = sycl::detail::lambda_arg_type<KernelType, item<Dims>>;
    static_assert(
        std::is_convertible_v<sycl::item<Dims>, LambdaArgType> ||
            std::is_convertible_v<sycl::item<Dims, false>, LambdaArgType>,
        "Kernel argument of a sycl::parallel_for with sycl::range "
        "must be either sycl::item or be convertible from sycl::item");
    using TranformedLambdaArgType = std::conditional_t<
        std::is_convertible_v<item<Dims>, LambdaArgType>, item<Dims>,
        std::conditional_t<
            std::is_convertible_v<item<Dims, false>, LambdaArgType>,
            item<Dims, false>, LambdaArgType>>;

    using NameT =
        typename detail::get_kernel_name_t<KernelName, KernelType>::name;
    submitParallelFor<NameT, TranformedLambdaArgType, KernelType>(rest...);
    return getLastEvent();
  }

  /// Name of this function is defined by compiler. It generates a call to this
  /// function in the host implementation of KernelFunc in submitSingleTask or
  /// submitParallelFor.
  /// \param KernelName the name of the kernel being invoked.
  /// \param args the kernel arguments for the kernel invocation.
  template <typename KN, typename... Args>
  void sycl_kernel_launch(const char *KernelName, Args &&...args) {
    static_assert(
        sizeof...(args) == 1,
        "sycl_kernel_launch expects only 2 arguments now: name of kernel and "
        "callable object passed to kernel invocation by the user.");

    auto FirstArg = std::get<0>(std::tie(args...));
    submitKernelImpl(detail::getDeviceKernelInfo<KN>(KernelName), &FirstArg,
                     sizeof(FirstArg));
  }

  /// The sycl_kernel_entry_point attribute facilitates the generation of an
  /// offload kernel entry point function with parameters corresponding to the
  /// (potentially decomposed) kernel arguments and a body that executes the
  /// kernel (after reconstructing the arguments if required).
#ifdef SYCL_LANGUAGE_VERSION
#  define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)                              \
    [[clang::sycl_kernel_entry_point(KernelName)]]
#else
#  define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
#endif // SYCL_LANGUAGE_VERSION

  /// Specifies the parameters and body of the generated offload kernel entry
  /// point for single_task invocations. On host, the compiler generates a call
  /// to sycl_kernel_launch instead of the KernelFunc invocation.
  template <typename KernelName, typename KernelType>
  _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
  void submitSingleTask(const KernelType &KernelFunc) {
    KernelFunc();
  }

  /// Specifies the parameters and body of the generated offload kernel entry
  /// point for parallel_for invocations. On host, the compiler generates a call
  /// to sycl_kernel_launch instead of the KernelFunc invocation.
  template <typename KernelName, typename ElementType, typename KernelType>
  _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
  void submitParallelFor(const KernelType &KernelFunc) {
    KernelFunc(detail::Builder::getElement(detail::declptr<ElementType>()));
  }
#undef _LIBSYCL_ENTRY_POINT_ATTR__

  /// Passes kernel parameters to the runtime.
  /// \param Events a collection of events representing dependencies of the
  /// kernel to submit.
  /// \param Range a unified view of the kernel execution range.
  void setKernelParameters(const std::vector<event> &Events,
                           const detail::UnifiedRangeView &Range = {});

  /// Passes kernel arguments to runtime.
  /// If all the dependencies can be handled by the backend, the kernel is
  /// submitted to it directly in this call.
  /// \param KernelInfo the information for the kernel being invoked.
  /// \param ArgData a pointer to the kernel argument.
  /// \param ArgSize the size of the kernel argument.
  void submitKernelImpl(detail::DeviceKernelInfo &KernelInfo, void *ArgData,
                        size_t ArgSize);

  /// \return an event representing last kernel invocation.
  event getLastEvent();

  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  friend sycl::detail::ImplUtils;
  friend sycl::detail::MockQueue;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
