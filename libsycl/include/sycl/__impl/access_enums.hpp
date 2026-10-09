//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the SYCL 2020 enumerations that describe an address
/// space and the decoration of a multi_ptr (4.7.7.).
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_ACCESS_ENUMS_HPP
#define _LIBSYCL___IMPL_ACCESS_ENUMS_HPP

#include <sycl/__impl/detail/config.hpp>

#include <cstdint>

_LIBSYCL_BEGIN_NAMESPACE_SYCL

namespace access {

enum class address_space : std::uint32_t {
  global_space = 0,
  local_space = 1,
  constant_space __SYCL2020_DEPRECATED(
      "the constant address space is deprecated in SYCL 2020") = 2,
  private_space = 3,
  generic_space = 4
};

enum class decorated : std::uint32_t { no = 0, yes = 1, legacy = 2 };

} // namespace access

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_ACCESS_ENUMS_HPP
