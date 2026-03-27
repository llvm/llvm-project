//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   struct tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

// UNSUPPORTED: c++03, c++11, c++14

#include <array>
#include <cassert>
#include <tuple>
#include <utility>

#include "test_macros.h"

struct S { int x; };

void test_decomp_user_type() {
  {
    S s{99};
    auto [m1] = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &s.x);
  }
  {
    S const s{99};
    auto [m1] = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &s.x);
  }
}

void test_decomp_tuple() {
  typedef std::tuple<int> T;
  {
    T s{99};
    auto [m1] = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &std::get<0>(s));
  }
  {
    T const s{99};
    auto [m1] = s;
    auto& [r1] = s;
    assert(m1 == 99);
    assert(&r1 == &std::get<0>(s));
  }
}


void test_decomp_pair() {
  typedef std::pair<int, double> T;
  {
    T s{99, 42.5};
    auto [m1, m2] = s;
    auto& [r1, r2] = s;
    assert(m1 == 99);
    assert(m2 == 42.5);
    assert(&r1 == &std::get<0>(s));
    assert(&r2 == &std::get<1>(s));
  }
  {
    T const s{99, 42.5};
    auto [m1, m2] = s;
    auto& [r1, r2] = s;
    assert(m1 == 99);
    assert(m2 == 42.5);
    assert(&r1 == &std::get<0>(s));
    assert(&r2 == &std::get<1>(s));
  }
}

void test_decomp_array() {
  typedef std::array<int, 3> T;
  {
    T s{{99, 42, -1}};
    auto [m1, m2, m3] = s;
    auto& [r1, r2, r3] = s;
    assert(m1 == 99);
    assert(m2 == 42);
    assert(m3 == -1);
    assert(&r1 == &std::get<0>(s));
    assert(&r2 == &std::get<1>(s));
    assert(&r3 == &std::get<2>(s));
  }
  {
    T const s{{99, 42, -1}};
    auto [m1, m2, m3] = s;
    auto& [r1, r2, r3] = s;
    assert(m1 == 99);
    assert(m2 == 42);
    assert(m3 == -1);
    assert(&r1 == &std::get<0>(s));
    assert(&r2 == &std::get<1>(s));
    assert(&r3 == &std::get<2>(s));
  }
}

struct TestLWG2770 {
  int n;
};

template <>
struct std::tuple_size<TestLWG2770> {};

void test_lwg_2770() {
  {
    auto [n] = TestLWG2770{42};
    assert(n == 42);
  }
  {
    const auto [n] = TestLWG2770{42};
    assert(n == 42);
  }
  {
    TestLWG2770 s{42};
    auto& [n] = s;
    assert(n == 42);
    assert(&n == &s.n);
  }
  {
    const TestLWG2770 s{42};
    auto& [n] = s;
    assert(n == 42);
    assert(&n == &s.n);
  }
}

struct Test {
  int x;
};

template <std::size_t N>
int get(Test const&) { static_assert(N == 0, ""); return -1; }

template <>
struct std::tuple_element<0, Test> {
  typedef int type;
};

void test_before_tuple_size_specialization() {
  Test const t{99};
  auto& [p] = t;
  assert(p == 99);
}

template <>
struct std::tuple_size<Test> {
public:
  static const std::size_t value = 1;
};

void test_after_tuple_size_specialization() {
  Test const t{99};
  auto& [p] = t;
  // https://cplusplus.github.io/LWG/issue4040
  // It is controversial whether std::tuple_size<const Test> is instantiated here or before.
  (void)p;
  LIBCPP_ASSERT(p == -1);
}

#if TEST_STD_VER >= 26 && __cpp_structured_bindings >= 202411L
struct InvalidWhenNoCv1 {};

template <>
struct std::tuple_size<InvalidWhenNoCv1> {};

struct InvalidWhenNoCv2 {};

template <>
struct std::tuple_size<InvalidWhenNoCv2> {
  void value();
};

template <class = void>
void test_decomp_as_empty_pack() {
  {
    const auto [... pack] = InvalidWhenNoCv1{};
    static_assert(sizeof...(pack) == 0);
  }
  {
    const auto [... pack] = InvalidWhenNoCv2{};
    static_assert(sizeof...(pack) == 0);
  }
}
#endif

int main(int, char**) {
  test_decomp_user_type();
  test_decomp_tuple();
  test_decomp_pair();
  test_decomp_array();
  test_lwg_2770();
  test_before_tuple_size_specialization();
  test_after_tuple_size_specialization();
#if TEST_STD_VER >= 26 && __cpp_structured_bindings >= 202411L
  test_decomp_as_empty_pack();
#endif

  return 0;
}
