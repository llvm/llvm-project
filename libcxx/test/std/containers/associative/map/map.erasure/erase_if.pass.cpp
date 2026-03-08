//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <map>

// template <class Key, class T, class Compare, class Allocator, class Predicate>
//   typename map<Key, T, Compare, Allocator>::size_type
//   constexpr erase_if(map<Key, T, Compare, Allocator>& c, Predicate pred); // constexpr since C++26

#include <map>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

using Init = std::initializer_list<int>;
template <typename M>
TEST_CONSTEXPR_CXX26 M make(Init vals) {
  M ret;
  for (int v : vals)
    ret[static_cast<typename M::key_type>(v)] = static_cast<typename M::mapped_type>(v + 10);
  return ret;
}

template <typename M, typename Pred>
TEST_CONSTEXPR_CXX26 void test0(Init vals, Pred p, Init expected, std::size_t expected_erased_count) {
  M s = make<M>(vals);
  ASSERT_SAME_TYPE(typename M::size_type, decltype(std::erase_if(s, p)));
  assert(expected_erased_count == std::erase_if(s, p));
  assert(s == make<M>(expected));
}

template <typename S>
TEST_CONSTEXPR_CXX26 bool test() {
  auto is1   = [](auto v) { return v.first == 1; };
  auto is2   = [](auto v) { return v.first == 2; };
  auto is3   = [](auto v) { return v.first == 3; };
  auto is4   = [](auto v) { return v.first == 4; };
  auto True  = [](auto) { return true; };
  auto False = [](auto) { return false; };

  test0<S>({}, is1, {}, 0);

  test0<S>({1}, is1, {}, 1);
  test0<S>({1}, is2, {1}, 0);

  test0<S>({1, 2}, is1, {2}, 1);
  test0<S>({1, 2}, is2, {1}, 1);
  test0<S>({1, 2}, is3, {1, 2}, 0);

  test0<S>({1, 2, 3}, is1, {2, 3}, 1);
  test0<S>({1, 2, 3}, is2, {1, 3}, 1);
  test0<S>({1, 2, 3}, is3, {1, 2}, 1);
  test0<S>({1, 2, 3}, is4, {1, 2, 3}, 0);

  test0<S>({1, 2, 3}, True, {}, 3);
  test0<S>({1, 2, 3}, False, {1, 2, 3}, 0);

  return true;
}

TEST_CONSTEXPR_CXX26
bool test_upper() {
  test<std::map<int, int>>();
  test<std::map<int, int, std::less<int>, min_allocator<std::pair<const int, int>>>>();
  test<std::map<int, int, std::less<int>, test_allocator<std::pair<const int, int>>>>();

  test<std::map<long, short>>();
  test<std::map<short, double>>();

  return true;
}

int main(int, char**) {
  assert(test_upper());

#if TEST_STD_VER >= 26
#  ifndef TEST_COMPILER_GCC
  // FIXME: Fails with g++-15 with:
  // clang-format off
  // __tree:116:23: error: ''result_decl' not supported by dump_expr<expression error>' is not a constant expression
  // clang-format on
  static_assert(test_upper());
#  endif
#endif

  return 0;
}
