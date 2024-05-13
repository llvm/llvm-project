//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template<class... TTypes, class... UTypes>
//   auto
//   operator<=>(const tuple<TTypes...>& t, const tuple<UTypes...>& u);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <tuple>

template <class T, class U>
concept can_compare = requires(T t, U u) { t <=> u; };

typedef std::tuple<int> T1;
typedef std::tuple<int, long> T2;

static_assert(!can_compare<T1, T2>);
static_assert(!can_compare<T2, T1>);
