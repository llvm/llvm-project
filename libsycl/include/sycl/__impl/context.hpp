//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL context class, which
/// represents the runtime data structures and state required by a SYCL backend
/// API to interact with a group of devices associated with a platform.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_CONTEXT_HPP
#define _LIBSYCL___IMPL_CONTEXT_HPP

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/info/desc_base.hpp>

#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>

#include <memory>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

class context;
class device;
class platform;

namespace detail {
class ContextImpl;
template <typename T>
using is_context_info_desc_t = typename is_info_desc<T, context>::return_type;
} // namespace detail

// SYCL 2020 4.6.3. Context class
class _LIBSYCL_EXPORT context {
public:
  context(const context &rhs) = default;

  context(context &&rhs) = default;

  context &operator=(const context &rhs) = default;

  context &operator=(context &&rhs) = default;

  friend bool operator==(const context &lhs, const context &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const context &lhs, const context &rhs) {
    return !(lhs == rhs);
  }

  /// Returns the backend associated with this context.
  ///
  /// \return the backend associated with this context.
  backend get_backend() const noexcept;

  /// Gets platform associated with this SYCL context.
  ///
  /// \return a valid instance of SYCL platform.
  platform get_platform() const;

  /// Gets devices associated with this SYCL context.
  ///
  /// \return a vector of valid SYCL device instances.
  std::vector<device> get_devices() const;

  /// Queries this SYCL context for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  detail::is_context_info_desc_t<Param> get_info() const;

  /// Queries this SYCL context for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

private:
  context(const std::shared_ptr<detail::ContextImpl> &Impl) : impl(Impl) {}
  std::shared_ptr<detail::ContextImpl> impl;

  friend sycl::detail::ImplUtils;
}; // class context

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::context> : public sycl::detail::HashBase<sycl::context> {
};

#endif // _LIBSYCL___IMPL_CONTEXT_HPP
