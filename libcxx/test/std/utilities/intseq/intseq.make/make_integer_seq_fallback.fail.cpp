//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

// UNSUPPORTED: c++98, c++03, c++11

// This test hangs during recursive template instantiation with libstdc++
// UNSUPPORTED: libstdc++

#define _LIBCPP_TESTING_FALLBACK_MAKE_INTEGER_SEQUENCE
#include "make_integer_seq.fail.cpp"
