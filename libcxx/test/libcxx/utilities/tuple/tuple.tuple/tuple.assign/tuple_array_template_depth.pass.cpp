//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <tuple>

// template <class... Types> class tuple;

// template <class Tuple, __tuple_assignable<Tuple, tuple> >
//   tuple & operator=(Tuple &&);

// This test checks that we do not evaluate __make_tuple_types
// on the array when it doesn't match the size of the tuple.

#include <array>
#include <tuple>

#include "test_macros.h"

// Use 1256 to try and blow the template instantiation depth for all compilers.
typedef std::array<char, 1256> array_t;
typedef std::tuple<array_t> tuple_t;

int main(int, char**)
{
    array_t arr;
    tuple_t tup;
    tup = arr;

  return 0;
}
