//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We don't know how to vectorize algorithms on GCC
// XFAIL: gcc

// We don't vectorize algorithms before C++14
// XFAIL: c++03, c++11

// We don't vectorize algorithms on AIX right now.
// XFAIL: target={{.+}}-aix{{.*}}

// This test ensures that we enable the vectorization of algorithms on the expected
// platforms.

#include <algorithm>

#ifndef _LIBCPP_VECTORIZE_ALGORITHMS
#  error It looks like the test needs to be updated since _LIBCPP_VECTORIZE_ALGORITHMS isn't defined anymore
#endif

#if !_LIBCPP_VECTORIZE_ALGORITHMS
#  error Algorithms should be vectorized on this platform
#endif
