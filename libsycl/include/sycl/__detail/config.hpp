//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the macros defining attributes for
/// exported methods and defining API namespaces.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_DETAIL_CONFIG_HPP
#define _LIBSYCL_DETAIL_CONFIG_HPP

#include <sycl/version.hpp>

#define _LIBSYCL_BEGIN_UNVERSIONED_NAMESPACE_SYCL namespace sycl {
#define _LIBSYCL_END_UNVERSIONED_NAMESPACE_SYCL }

#define _LIBSYCL_BEGIN_NAMESPACE_SYCL                                          \
  _LIBSYCL_BEGIN_UNVERSIONED_NAMESPACE_SYCL inline namespace _LIBSYCL_ABI_NAMESPACE {
#define _LIBSYCL_END_NAMESPACE_SYCL                                            \
  }                                                                            \
  _LIBSYCL_END_UNVERSIONED_NAMESPACE_SYCL

#ifndef __SYCL_DEVICE_ONLY__

#  ifndef _LIBSYCL_EXPORT
#    ifdef _WIN32

#      define _LIBSYCL_DLL_LOCAL

#      if _LIBSYCL_BUILD_SYCL_DLL
#        define _LIBSYCL_EXPORT __declspec(dllexport)
#      else
#        define _LIBSYCL_EXPORT __declspec(dllimport)
#      endif //_LIBSYCL_BUILD_SYCL_DLL

#    else // _WIN32

#      define _LIBSYCL_DLL_LOCAL [[gnu::visibility("hidden")]]
#      define _LIBSYCL_EXPORT [[gnu::visibility("default")]]

#    endif // _WIN32
#  endif   // _LIBSYCL_EXPORT

#else // __SYCL_DEVICE_ONLY__

#  ifndef _LIBSYCL_EXPORT
#    define _LIBSYCL_EXPORT
#    define _LIBSYCL_DLL_LOCAL
#  endif

#endif // __SYCL_DEVICE_ONLY__

#endif // _LIBSYCL_DETAIL_CONFIG_HPP
