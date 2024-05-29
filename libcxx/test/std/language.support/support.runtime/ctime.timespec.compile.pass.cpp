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

// picolibc doesn't define TIME_UTC.
// XFAIL: LIBCXX-PICOLIBC-FIXME

// ::timespec_get is provided by the C library, but it's marked as
// unavailable until macOS 10.15
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// ::timespec_get is available starting with Android Q (API 29)
// XFAIL: target={{.+}}-android{{(eabi)?(21|22|23|24|25|26|27|28)}}

// ::timespec_get is available starting with AIX 7.3 TL2
// XFAIL: target={{.+}}-aix{{7.2.*|7.3.0.*|7.3.1.*}}

#include <ctime>
#include <type_traits>

#ifndef TIME_UTC
#error TIME_UTC not defined
#endif

std::timespec tmspec = {};
static_assert(std::is_same<decltype(std::timespec_get(&tmspec, 0)), int>::value, "");
