//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains macros for suppressing extra warnings enabled in libsycl
/// for LLVM headers.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_SUPPRESS_EXTRA_WARNINGS
#define _LIBSYCL_SUPPRESS_EXTRA_WARNINGS

#define _LIBSYCL_DO_PRAGMA(x) _Pragma(#x)
#define _LIBSYCL_SUPPRESS_EXTRA_WARNINGS_BEGIN                                 \
  _LIBSYCL_DO_PRAGMA(GCC diagnostic push)                                      \
  _LIBSYCL_DO_PRAGMA(GCC diagnostic ignored "-Wshadow")
#define _LIBSYCL_SUPPRESS_EXTRA_WARNINGS_END                                   \
  _LIBSYCL_DO_PRAGMA(GCC diagnostic pop)

#endif // _LIBSYCL_SUPPRESS_EXTRA_WARNINGS
