//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that passing `-fexperimental-library` results in experimental
// library features being enabled.

// GCC does not support the -fexperimental-library flag
// UNSUPPORTED: gcc

// ADDITIONAL_COMPILE_FLAGS: -fexperimental-library

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <version>

#if !_LIBCPP_HAS_EXPERIMENTAL_PSTL
#  error "-fexperimental-library should enable the PSTL"
#endif

#if !_LIBCPP_HAS_EXPERIMENTAL_TZDB
#  error "-fexperimental-library should enable the chrono TZDB"
#endif

#if !_LIBCPP_HAS_EXPERIMENTAL_SYNCSTREAM
#  error "-fexperimental-library should enable the syncstream header"
#endif
