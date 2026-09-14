//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the SYCL enum class backend that is
/// implementation-defined and is populated with a unique identifier for each
/// SYCL backend that the SYCL implementation can support.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_BACKEND_HPP
#define _LIBSYCL___IMPL_BACKEND_HPP

#include <sycl/__impl/detail/config.hpp>

#include <string_view>
#include <type_traits>

// SYCL 2020 4.1.1. Backend macros.
#define SYCL_EXT_LLVM_BACKEND_LEVEL_ZERO 1
// The macros for backend::opencl, backend::ext_llvm_cuda and
// backend::ext_llvm_hip are intentionally left undefined, as these backends are
// not available yet.

_LIBSYCL_BEGIN_NAMESPACE_SYCL

// SYCL 2020 4.1. Backends and SYCL 2020 6.3.7. Adding a backend.
enum class backend : unsigned char {
  opencl = 0,
  ext_llvm_level_zero,
  ext_llvm_cuda,
  ext_llvm_hip,
};

namespace detail {
template <typename T> struct is_backend_info_desc : std::false_type {};
} // namespace detail

// SYCL 2020  4.5.1.1. Type traits backend_traits.
template <backend Backend> class backend_traits;

template <backend Backend, typename SyclType>
using backend_input_t =
    typename backend_traits<Backend>::template input_type<SyclType>;
template <backend Backend, typename SyclType>
using backend_return_t =
    typename backend_traits<Backend>::template return_type<SyclType>;

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL___IMPL_BACKEND_HPP
