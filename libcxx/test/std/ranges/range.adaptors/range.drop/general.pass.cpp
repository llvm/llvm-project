//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Some basic examples of how drop_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <ranges>

#include <vector>
#include <list>
#include <string>

#include <cassert>
#include "test_macros.h"
#include "test_iterators.h"
#include "types.h"

template<class T>
concept ValidDropView = requires { typename std::ranges::drop_view<T>; };

static_assert( ValidDropView<MoveOnlyView>);
static_assert(!ValidDropView<Range>);

static_assert(!std::ranges::enable_borrowed_range<std::ranges::drop_view<MoveOnlyView>>);
static_assert( std::ranges::enable_borrowed_range<std::ranges::drop_view<BorrowableView>>);

template<std::ranges::view View>
bool orderedFibonacci(View v, int n = 1) {
  if (v.size() < 3)
    return true;

  if (v[2] != v[0] + v[1])
    return false;

  return orderedFibonacci(std::ranges::drop_view(v.base(), n), n + 1);
}

template<std::ranges::view View>
std::ranges::view auto makeEven(View v) {
  return std::ranges::drop_view(v, v.size() % 2);
}

template<std::ranges::view View, class T>
int indexOf(View v, T element) {
  int index = 0;
  for (auto e : v) {
    if (e == element)
      return index;
    index++;
  }
  return -1;
}

template<std::ranges::view View, class T>
std::ranges::view auto removeBefore(View v, T element) {
  std::ranges::drop_view out(v, indexOf(v, element) + 1);
  return View(out.begin(), out.end());
}

template<>
constexpr bool std::ranges::enable_view<std::vector<int>> = true;

template<>
constexpr bool std::ranges::enable_view<std::list<int>> = true;

template<>
constexpr bool std::ranges::enable_view<std::string> = true;

int main(int, char**) {
  const std::vector vec = {1,1,2,3,5,8,13};
  assert(orderedFibonacci(std::ranges::drop_view(vec, 0)));
  const std::vector vec2 = {1,1,2,3,5,8,14};
  assert(!orderedFibonacci(std::ranges::drop_view(vec2, 0)));

  const std::list l = {1, 2, 3};
  auto el = makeEven(l);
  assert(el.size() == 2);
  assert(*el.begin() == 2);

  const std::string s = "Hello, World";
  assert(removeBefore(s, ' ') == "World");

  return 0;
}
