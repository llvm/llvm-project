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
/// exported methods.
///
//===----------------------------------------------------------------------===//

#ifndef __LIBSYCL_DETAIL_EXPORT_HPP
#define __LIBSYCL_DETAIL_EXPORT_HPP

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

#endif // __LIBSYCL_DETAIL_EXPORT_HPP