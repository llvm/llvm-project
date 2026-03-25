//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL event class (SYCL
/// 2020 4.6.6.), that represents the status of an operation that is being
/// executed by the SYCL runtime.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_EVENT_HPP
#define _LIBSYCL___IMPL_EVENT_HPP

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/info/desc_base.hpp>

#include <memory>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class event;

namespace detail {
class EventImpl;
template <typename T>
using is_event_info_desc_t = typename is_info_desc<T, event>::return_type;
} // namespace detail

/// SYCL 2020 4.6.6. Event class.
class _LIBSYCL_EXPORT event {
public:
  event(const event &rhs) = default;

  event(event &&rhs) = default;

  event &operator=(const event &rhs) = default;

  event &operator=(event &&rhs) = default;

  friend bool operator==(const event &lhs, const event &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const event &lhs, const event &rhs) {
    return !(lhs == rhs);
  }

  /// \return the backend associated with this platform.
  backend get_backend() const noexcept;

  /// Blocks until all commands associated with this event and any dependent
  /// events have completed.
  void wait();

  /// Behaves as if calling event::wait on each event in eventList.
  static void wait(const std::vector<event> &eventList);

  /// Queries this SYCL event for information.
  ///
  /// \return depends on the information being requested.
  template <typename Param>
  detail::is_event_info_desc_t<Param> get_info() const;

  /// Queries this SYCL event for SYCL backend-specific information.
  ///
  /// \return depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  event(std::shared_ptr<detail::EventImpl> Impl) : impl(Impl) {}
  std::shared_ptr<detail::EventImpl> impl;

  friend sycl::detail::ImplUtils;
};

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::event> : public sycl::detail::HashBase<sycl::event> {};

#endif // _LIBSYCL___IMPL_EVENT_HPP
