//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the SYCL 2020 enumerations that describe how an accessor
/// accesses a memory object: access_mode (4.7.6.2.), target (4.7.6.1.) and the
/// deprecated access::placeholder.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ACCESS_ENUMS_HPP
#define _LIBSYCL___IMPL_ACCESS_ENUMS_HPP

#include <sycl/__impl/detail/config.hpp>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

enum class access_mode {
  read = 0,
  write = 1,
  read_write = 2,
  discard_write __SYCL2020_DEPRECATED(
      "use 'write' with the 'no_init' property instead") = 3,
  discard_read_write __SYCL2020_DEPRECATED(
      "use 'read_write' with the 'no_init' property instead") = 4,
  atomic __SYCL2020_DEPRECATED("use 'atomic_ref' instead") = 5
};

enum class target {
  device = 0,
  host_task = 1,
  constant_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = 2,
  local __SYCL2020_DEPRECATED("use 'local_accessor' instead") = 3,
  host_buffer __SYCL2020_DEPRECATED("use 'host_accessor' instead") = 4,
  global_buffer __SYCL2020_DEPRECATED("use 'target::device' instead") = device
};

namespace access {

using mode __SYCL2020_DEPRECATED("use 'sycl::access_mode' instead") =
    sycl::access_mode;

using target __SYCL2020_DEPRECATED("use 'sycl::target' instead") = sycl::target;

enum class __SYCL2020_DEPRECATED(
    "placeholder accessors are deprecated in SYCL 2020") placeholder {
  false_t = 0,
  true_t = 1
};

} // namespace access

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ACCESS_ENUMS_HPP
