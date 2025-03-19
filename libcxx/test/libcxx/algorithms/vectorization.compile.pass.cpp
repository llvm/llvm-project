//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// We don't know how to vectorize algorithms on GCC
// XFAIL: gcc

// This test ensures that we enable the vectorization of algorithms on the expected
// platforms.

#include <algorithm>

#if !_LIBCPP_VECTORIZE_ALGORITHMS
#  error Algorithms should be vectorized on this platform
#endif
