//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the macroses defining attributes for
/// exported methods and defining API namespaces.
///
//===----------------------------------------------------------------------===//

#ifndef __LIBSYCL_DETAIL_CONFIG_HPP
#define __LIBSYCL_DETAIL_CONFIG_HPP

#include <sycl/version.hpp>

#define __SYCL_BEGIN_UNVERSIONED_NAMESPACE namespace sycl {
#define __SYCL_END_UNVERSIONED_NAMESPACE }

#define __SYCL_BEGIN_VERSIONED_NAMESPACE                                       \
  __SYCL_BEGIN_UNVERSIONED_NAMESPACE inline namespace __LIBSYCL_ABI_NAMESPACE {
#define __SYCL_END_VERSIONED_NAMESPACE                                         \
  }                                                                            \
  __SYCL_END_UNVERSIONED_NAMESPACE

#ifndef __SYCL_DEVICE_ONLY__
#ifndef __SYCL_EXPORT
#ifdef _WIN32

#define __SYCL_DLL_LOCAL

#if __SYCL_BUILD_SYCL_DLL
#define __SYCL_EXPORT __declspec(dllexport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllexport, deprecated(x))
#else
#define __SYCL_EXPORT __declspec(dllimport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllimport, deprecated(x))
#endif //__SYCL_BUILD_SYCL_DLL

#else // _WIN32

#define __SYCL_DLL_LOCAL __attribute__((visibility("hidden")))

#define __SYCL_EXPORT __attribute__((visibility("default")))
#define __SYCL_EXPORT_DEPRECATED(x)                                            \
  __attribute__((visibility("default"), deprecated(x)))

#endif // _WIN32
#endif // __SYCL_EXPORT

#else // __SYCL_DEVICE_ONLY__

#ifndef __SYCL_EXPORT
#define __SYCL_EXPORT
#define __SYCL_EXPORT_DEPRECATED(x)
#define __SYCL_DLL_LOCAL
#endif

#endif // __SYCL_DEVICE_ONLY__

#endif // __LIBSYCL_DETAIL_CONFIG_HPP