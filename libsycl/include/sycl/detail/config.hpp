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

#ifndef __LIBSYCL_DETAIL_CONFIG_HPP
#define __LIBSYCL_DETAIL_CONFIG_HPP

#include <sycl/version.hpp>

#define __LIBSYCL_BEGIN_UNVERSIONED_NAMESPACE namespace sycl {
#define __LIBSYCL_END_UNVERSIONED_NAMESPACE }

#define __LIBSYCL_BEGIN_VERSIONED_NAMESPACE                                    \
  __LIBSYCL_BEGIN_UNVERSIONED_NAMESPACE inline namespace __LIBSYCL_ABI_NAMESPACE {
#define __LIBSYCL_END_VERSIONED_NAMESPACE                                      \
  }                                                                            \
  __LIBSYCL_END_UNVERSIONED_NAMESPACE

#ifndef __SYCL_DEVICE_ONLY__
#ifndef __LIBSYCL_EXPORT
#ifdef _WIN32

#define __LIBSYCL_DLL_LOCAL

#if __LIBSYCL_BUILD_SYCL_DLL
#define __LIBSYCL_EXPORT __declspec(dllexport)
#define __LIBSYCL_EXPORT_DEPRECATED(x) __declspec(dllexport, deprecated(x))
#else
#define __LIBSYCL_EXPORT __declspec(dllimport)
#define __LIBSYCL_EXPORT_DEPRECATED(x) __declspec(dllimport, deprecated(x))
#endif //__LIBSYCL_BUILD_SYCL_DLL

#else // _WIN32

#define __LIBSYCL_DLL_LOCAL __attribute__((visibility("hidden")))

#define __LIBSYCL_EXPORT __attribute__((visibility("default")))
#define __LIBSYCL_EXPORT_DEPRECATED(x)                                         \
  __attribute__((visibility("default"), deprecated(x)))

#endif // _WIN32
#endif // __LIBSYCL_EXPORT

#else // __SYCL_DEVICE_ONLY__

#ifndef __LIBSYCL_EXPORT
#define __LIBSYCL_EXPORT
#define __LIBSYCL_EXPORT_DEPRECATED(x)
#define __LIBSYCL_DLL_LOCAL
#endif

#endif // __SYCL_DEVICE_ONLY__

#endif // __LIBSYCL_DETAIL_CONFIG_HPP
