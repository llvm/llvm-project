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

// Clang does not support the -fexperimental-library flag before LLVM 15.0
// UNSUPPORTED: clang-13, clang-14

// AppleClang does not support the -fexperimental-library flag yet
// UNSUPPORTED: apple-clang-13, apple-clang-14.0

// Clang on AIX currently pretends that it is Clang 15, even though it is not (as of writing
// this, LLVM 15 hasn't even been branched yet).
// UNSUPPORTED: clang-15 && buildhost=aix

// ADDITIONAL_COMPILE_FLAGS: -fexperimental-library

#include <version>

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_FORMAT
#   error "-fexperimental-library should enable <format>"
#endif

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_RANGES
#   error "-fexperimental-library should enable <ranges>"
#endif
