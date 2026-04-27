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
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/get_device_kernel_info.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/detail/unified_range_view.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;

namespace detail {
class QueueImpl;

template <typename, typename T> struct CheckFunctionSignature {
  static_assert(std::integral_constant<T, false>::value,
                "Second template parameter is required to be of function type");
};

template <typename F, typename RetT, typename... Args>
struct CheckFunctionSignature<F, RetT(Args...)> {
private:
  template <typename T>
  static constexpr auto check(T *) -> typename std::is_same<
      decltype(std::declval<T>().operator()(std::declval<Args>()...)),
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

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName, typename KernelType>
  event single_task(const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>({}, kernelFunc);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param depEvent is an event that specifies the kernel dependency.
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName, typename KernelType>
  event single_task(event depEvent, const KernelType &kernelFunc) {
    return single_task<KernelName, KernelType>({depEvent}, kernelFunc);
  }

  /// Defines and invokes a SYCL kernel function as a lambda expression or a
  /// named function object type.
  ///
  /// \param depEvents is a collection of events that specify the kernel
  /// dependencies.
  /// \param kernelFunc is the kernel functor or lambda.
  /// \return an event that represents the status of the submitted kernel.
  template <typename KernelName, typename KernelType>
  event single_task(const std::vector<event> &depEvents,
                    const KernelType &kernelFunc) {
    static_assert(
        detail::CheckFunctionSignature<std::remove_reference_t<KernelType>,
                                       void()>::value,
        "sycl::queue::single_task() requires a kernel instead of a command "
        "group");

    setKernelParameters(depEvents);
    submitSingleTask<KernelName, KernelType>(kernelFunc);
    return getLastEvent();
  }

private:
  // Name of this function is defined by compiler. It generates call to this
  // function in the host implementation of KernelFunc in submitSingleTask.
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

#ifdef SYCL_LANGUAGE_VERSION
#  define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)                              \
    [[clang::sycl_kernel_entry_point(KernelName)]]
#else
#  define _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
#endif // SYCL_LANGUAGE_VERSION

  template <typename KernelName, typename KernelType>
  _LIBSYCL_ENTRY_POINT_ATTR__(KernelName)
  void submitSingleTask(const KernelType &KernelFunc) {
    KernelFunc();
  }
#undef _LIBSYCL_ENTRY_POINT_ATTR__

  event getLastEvent();
  void submitKernelImpl(detail::DeviceKernelInfo &KernelInfo, void *ArgData,
                        size_t ArgSize);
  void setKernelParameters(const std::vector<event> &Events,
                           const detail::UnifiedRangeView &Range = {});

  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  friend sycl::detail::ImplUtils;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
