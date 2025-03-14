//==---------------- export.hpp - SYCL standard header file ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef __SYCL_DEVICE_ONLY__
#ifndef __SYCL_EXPORT
#ifdef _WIN32

// MSVC discourages export of classes, that use STL class in API. This
// results in a warning, treated as compile error. Silence C4251 to workaround.
#pragma warning(disable : 4251)
#pragma warning(disable : 4275)

#define __SYCL_DLL_LOCAL

#if __SYCL_BUILD_SYCL_DLL
#define __SYCL_EXPORT __declspec(dllexport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllexport, deprecated(x))
#else
#define __SYCL_EXPORT __declspec(dllimport)
#define __SYCL_EXPORT_DEPRECATED(x) __declspec(dllimport, deprecated(x))
#endif //__SYCL_BUILD_SYCL_DLL
#else  // _WIN32

#define __SYCL_DLL_LOCAL __attribute__((visibility("hidden")))

#define __SYCL_EXPORT __attribute__((visibility("default")))
#define __SYCL_EXPORT_DEPRECATED(x)                                            \
  __attribute__((visibility("default"), deprecated(x)))
#endif // _WIN32
#endif // __SYCL_EXPORT
#else
#ifndef __SYCL_EXPORT
#define __SYCL_EXPORT
#define __SYCL_EXPORT_DEPRECATED(x)
#define __SYCL_DLL_LOCAL
#endif
#endif // __SYCL_DEVICE_ONLY__
