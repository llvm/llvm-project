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

// Clang on AIX currently pretends that it is Clang 15, even though it is not (as of writing
// this, LLVM 15 hasn't even been branched yet).
// UNSUPPORTED: clang-15 && buildhost=aix

// ADDITIONAL_COMPILE_FLAGS: -fexperimental-library

#include <version>

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
#  error "-fexperimental-library should enable the PSTL"
#endif

#ifdef _LIBCPP_HAS_NO_EXPERIMENTAL_STOP_TOKEN
#  error "-fexperimental-library should enable the stop_token"
#endif

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_TZDB
#  error "-fexperimental-library should enable the chrono TZDB"
#endif
