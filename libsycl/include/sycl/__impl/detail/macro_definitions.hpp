//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains macro definitions used in SYCL implementation.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL___IMPL_DETAIL_MACRO_DEFINITIONS_HPP
#define _LIBSYCL___IMPL_DETAIL_MACRO_DEFINITIONS_HPP

static_assert(__cplusplus >= 201703L, "Libsycl requires C++17 or later.");

#ifndef __SYCL2020_DEPRECATED
#  if SYCL_LANGUAGE_VERSION == 202012L &&                                      \
      !defined(SYCL2020_DISABLE_DEPRECATION_WARNINGS)
#    define __SYCL2020_DEPRECATED(message) [[deprecated(message)]]
#  else
#    define __SYCL2020_DEPRECATED(message)
#  endif
#endif // __SYCL2020_DEPRECATED

#if defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)
// When built for use with the MSVC C++ standard library, libsycl requires
// use of the DLL versions of the MSVC run-time (RT) library. This requirement
// extends to applications that link with libsycl since the same MSVC run-time
// library must be used to ensure ABI compatibility for objects of C++ standard
// library types like std::vector that are passed to or returned from SYCL
// interfaces. Applications must therefore compile and link with the /MD option
// when linking to a release build of libsycl and with the /MDd option when
// linking to a debug build.
#  define ERROR_MESSAGE                                                        \
    "Libsycl requires use of a DLL version of the MSVC RT library. "           \
    "Please use /MD to link with a release build of libsycl or /MDd to link"   \
    " with a debug build."
#  if defined(_MSC_VER)
#    pragma message(ERROR_MESSAGE)
#  else
#    warning ERROR_MESSAGE
#  endif
#  undef ERROR_MESSAGE
#endif // defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)

#endif //_LIBSYCL___IMPL_DETAIL_MACRO_DEFINITIONS_HPP
