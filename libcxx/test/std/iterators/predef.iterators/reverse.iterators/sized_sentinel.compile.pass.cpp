//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// reverse_iterator

#include <iterator>

#include "test_iterators.h"

template<class T> concept HasMinus = requires (T t) { t - t; };

using sized_it = random_access_iterator<int*>;
static_assert( std::sized_sentinel_for<sized_it, sized_it>);
static_assert( std::sized_sentinel_for<std::reverse_iterator<sized_it>, std::reverse_iterator<sized_it>>);
static_assert( HasMinus<std::reverse_iterator<sized_it>>);

// Check that `sized_sentinel_for` is false for `reverse_iterator`s if it is false for the underlying iterators.
using unsized_it = bidirectional_iterator<int*>;
static_assert(!std::sized_sentinel_for<unsized_it, unsized_it>);
static_assert(!std::sized_sentinel_for<std::reverse_iterator<unsized_it>, std::reverse_iterator<unsized_it>>);
static_assert(!HasMinus<std::reverse_iterator<unsized_it>>);
