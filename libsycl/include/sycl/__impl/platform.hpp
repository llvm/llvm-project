//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL platform class, which
/// encapsulates a single platform on which kernel functions may be executed.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_PLATFORM_HPP
#define _LIBSYCL___IMPL_PLATFORM_HPP

#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/detail/config.hpp>
#include <sycl/__impl/detail/obj_utils.hpp>
#include <sycl/__impl/info/platform.hpp>

#include <memory>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace detail {
class PlatformImpl;
} // namespace detail

/// \brief SYCL 2020 platform class (4.6.2.) encapsulating a single SYCL
/// platform on which kernel functions may be executed.
class _LIBSYCL_EXPORT platform {
public:
  // The platform class provides the common reference semantics (SYCL
  // 2020 4.5.2).
  platform(const platform &rhs) = default;

  platform(platform &&rhs) = default;

  platform &operator=(const platform &rhs) = default;

  platform &operator=(platform &&rhs) = default;

  friend bool operator==(const platform &lhs, const platform &rhs) {
    return lhs.impl == rhs.impl;
  }

  friend bool operator!=(const platform &lhs, const platform &rhs) {
    return !(lhs == rhs);
  }

  /// Returns the backend associated with this platform.
  ///
  /// \return the backend associated with this platform.
  backend get_backend() const noexcept;

  /// Queries this SYCL platform for info.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  detail::is_platform_info_desc_t<Param> get_info() const;

  /// Queries this SYCL platform for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename detail::is_backend_info_desc<Param>::return_type
  get_backend_info() const;

  /// Returns all SYCL platforms from all backends that are available in the
  /// system.
  ///
  /// \return A std::vector containing all of the platforms from all backends
  /// that are available in the system.
  static std::vector<platform> get_platforms();

private:
  platform(detail::PlatformImpl &Impl) : impl(&Impl) {}
  detail::PlatformImpl *impl;

  friend sycl::detail::ImplUtils;
}; // class platform

_LIBSYCL_END_NAMESPACE_SYCL

template <>
struct std::hash<sycl::platform>
    : public sycl::detail::HashBase<sycl::platform> {};

#endif // _LIBSYCL___IMPL_PLATFORM_HPP
