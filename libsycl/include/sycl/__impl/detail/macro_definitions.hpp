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

#ifndef __SYCL2020_DEPRECATED
#  if SYCL_LANGUAGE_VERSION == 202012L &&                                      \
      !defined(SYCL2020_DISABLE_DEPRECATION_WARNINGS)
#    define __SYCL2020_DEPRECATED(message) [[deprecated(message)]]
#  else
#    define __SYCL2020_DEPRECATED(message)
#  endif
#endif // __SYCL2020_DEPRECATED

static_assert(__cplusplus >= 201703L,
              "SYCL RT does not support C++ version earlier than C++17.");

#if defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)
// SYCL library is designed such a way that STL objects cross DLL boundary,
// which is guaranteed to work properly only when the application uses the same
// C++ runtime that SYCL library uses.
// The appplications using sycl.dll must be linked with dynamic/release C++ MSVC
// runtime, i.e. be compiled with /MD switch. Similarly, the applications using
// sycld.dll must be linked with dynamic/debug C++ runtime and be compiled with
// /MDd switch.
// Compiler automatically adds /MD or /MDd when -fsycl switch is used.
// The options /MD and /MDd that make the code to use dynamic runtime also
// define the _DLL macro.
#  define ERROR_MESSAGE                                                        \
    "SYCL library is designed to work safely with dynamic C++ runtime."        \
    "Please use /MD switch with sycl.dll, /MDd switch with sycld.dll, "        \
    "or -fsycl switch to set C++ runtime automatically."
#  if defined(_MSC_VER)
#    pragma message(ERROR_MESSAGE)
#  else
#    warning ERROR_MESSAGE
#  endif
#  undef ERROR_MESSAGE
#endif // defined(_WIN32) && !defined(_DLL) && !defined(__SYCL_DEVICE_ONLY__)

#endif //_LIBSYCL___IMPL_DETAIL_MACRO_DEFINITIONS_HPP
