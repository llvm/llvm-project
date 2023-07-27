//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-aligned_allocation-missing

// https://reviews.llvm.org/D129198 is not in AppleClang 14
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.13{{(.0)?}} && apple-clang-14

#include <new>

#include "test_macros.h"


#ifdef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
#   error "libc++ should have aligned allocation in C++17 and up when targeting a platform that supports it"
#endif
