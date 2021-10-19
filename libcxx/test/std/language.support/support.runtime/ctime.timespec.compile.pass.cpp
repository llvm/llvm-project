//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test <ctime>
// std::timespec and std::timespec_get

// UNSUPPORTED: c++03, c++11, c++14

// ::timespec_get is provided by the C library, but it's marked as
// unavailable until macOS 10.15
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

#include <ctime>
#include <type_traits>

#ifndef TIME_UTC
#error TIME_UTC not defined
#endif

std::timespec tmspec = {};
static_assert(std::is_same<decltype(std::timespec_get(&tmspec, 0)), int>::value, "");
