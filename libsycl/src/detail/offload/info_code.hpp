//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_INFO_CODE
#define _LIBSYCL_INFO_CODE

_LIBSYCL_BEGIN_NAMESPACE_SYCL

#include <OffloadAPI.h>

namespace detail {
template <typename T> struct OffloadInfoCode;

#define __SYCL_PARAM_TRAITS_SPEC(DescType, Desc, ReturnT, OffloadCode)         \
  template <> struct OffloadInfoCode<info::DescType::Desc> {                   \
    static constexpr auto value = OffloadCode;                                 \
  };
#include <sycl/__impl/info/platform.def>
#undef __SYCL_PARAM_TRAITS_SPEC

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_INFO_CODE
