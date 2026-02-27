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

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/default_async_handler.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <sycl/__impl/async_handler.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/property_list.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;

namespace detail {
class QueueImpl;
} // namespace detail

// SYCL 2020 4.6.5. Queue class.
class _LIBSYCL_EXPORT queue {
public:
  queue(const queue &rhs) = default;

  queue(queue &&rhs) = default;

  queue &operator=(const queue &rhs) = default;

  queue &operator=(queue &&rhs) = default;

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
  /// \param deviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
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
  /// \param deviceSelector is SYCL 2020 Device Selector, a simple callable that
  /// takes a device and returns an int
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

  /// Returns the SYCL backend that is associated with this queue.
  ///
  /// \return the backend associated with this queue.
  backend get_backend() const noexcept;

  /// Returns context that is associated with this queue.
  ///
  /// \return an associated SYCL context.
  context get_context() const;

  /// Returns device that is associated with this queue.
  ///
  /// \return SYCL device this queue was constructed with.
  device get_device() const;

  /// Returns whether the queue is in order or out of order.
  ///
  /// Equivalent to has_property<property::queue::in_order>().
  ///
  /// \return true if queue is in order.
  bool is_in_order() const;

  /// Queries SYCL queue for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  queue(const std::shared_ptr<detail::QueueImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::QueueImpl> impl;

  friend sycl::detail::ImplUtils;
}; // class queue

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::queue> : public sycl::detail::HashBase<sycl::queue> {};

#endif // _LIBSYCL___IMPL_QUEUE_HPP
