//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "type_algorithms.h"

// concatenate
static_assert(std::is_same<types::concatenate_t<types::type_list<> >, types::type_list<> >::value, "");
static_assert(std::is_same<types::concatenate_t<types::type_list<int> >, types::type_list<int> >::value, "");
static_assert(
    std::is_same<types::concatenate_t<types::type_list<int>, types::type_list<long> >, types::type_list<int, long> >::value,
    "");
static_assert(
    std::is_same<types::concatenate_t<types::type_list<int>, types::type_list<long>, types::type_list<long long> >,
                 types::type_list<int, long, long long> >::value,
    "");

// apply_all
template <int N>
class NumT {};

struct ApplyAllTest {
  bool* is_called_array_;

  TEST_CONSTEXPR ApplyAllTest(bool* is_called_array) : is_called_array_(is_called_array) {}

  template <int N>
  TEST_CONSTEXPR_CXX20 void check_num(NumT<N>) {
    assert(!is_called_array_[N]);
    is_called_array_[N] = true;
  }

  template <int N, int M>
  TEST_CONSTEXPR_CXX20 void check_num(NumT<N>, NumT<M>) {
    assert(!is_called_array_[N + M]);
    is_called_array_[N + M] = true;
  }

  template <class... Types>
  TEST_CONSTEXPR_CXX20 void operator()() {
    check_num(Types()...);
  }
};

struct Identity {
  TEST_CONSTEXPR bool operator()(bool b) const { return b; }
};

TEST_CONSTEXPR_CXX20 void test_for_each() {
  bool is_called_array[3] = {};
  types::for_each(types::type_list<NumT<0>, NumT<1>, NumT<2> >(), ApplyAllTest(is_called_array));
  assert(std::all_of(is_called_array, is_called_array + 3, Identity()));
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_for_each();
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
