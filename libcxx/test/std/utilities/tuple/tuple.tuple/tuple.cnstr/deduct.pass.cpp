//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// UNSUPPORTED: apple-clang-9

// GCC's implementation of class template deduction is still immature and runs
// into issues with libc++. However GCC accepts this code when compiling
// against libstdc++.
// XFAIL: gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10, gcc-11

// <tuple>

// Test that the constructors offered by std::tuple are formulated
// so they're compatible with implicit deduction guides, or if that's not
// possible that they provide explicit guides to make it work.

#include <tuple>
#include <cassert>
#include <functional>
#include <memory>

#include "test_macros.h"
#include "archetypes.h"


// Overloads
//  using A = Allocator
//  using AT = std::allocator_arg_t
// ---------------
// (1)  tuple(const Types&...) -> tuple<Types...>
// (2)  tuple(pair<T1, T2>) -> tuple<T1, T2>;
// (3)  explicit tuple(const Types&...) -> tuple<Types...>
// (4)  tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (5)  explicit tuple(AT, A const&, Types const&...) -> tuple<Types...>
// (6)  tuple(AT, A, pair<T1, T2>) -> tuple<T1, T2>
// (7)  tuple(tuple const& t) -> decltype(t)
// (8)  tuple(tuple&& t) -> decltype(t)
// (9)  tuple(AT, A const&, tuple const& t) -> decltype(t)
// (10) tuple(AT, A const&, tuple&& t) -> decltype(t)
void test_primary_template()
{
  const std::allocator<int> A;
  const auto AT = std::allocator_arg;
  { // Testing (1)
    int x = 101;
    std::tuple t1(42);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<int>);
    std::tuple t2(x, 0.0, nullptr);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<int, double, decltype(nullptr)>);
  }
  { // Testing (2)
    std::pair<int, char> p1(1, 'c');
    std::tuple t1(p1);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<int, char>);

    std::pair<int, std::tuple<char, long, void*>> p2(1, std::tuple<char, long, void*>('c', 3l, nullptr));
    std::tuple t2(p2);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<int, std::tuple<char, long, void*>>);

    int i = 3;
    std::pair<std::reference_wrapper<int>, char> p3(std::ref(i), 'c');
    std::tuple t3(p3);
    ASSERT_SAME_TYPE(decltype(t3), std::tuple<std::reference_wrapper<int>, char>);

    std::pair<int&, char> p4(i, 'c');
    std::tuple t4(p4);
    ASSERT_SAME_TYPE(decltype(t4), std::tuple<int&, char>);

    std::tuple t5(std::pair<int, char>(1, 'c'));
    ASSERT_SAME_TYPE(decltype(t5), std::tuple<int, char>);
  }
  { // Testing (3)
    using T = ExplicitTestTypes::TestType;
    static_assert(!std::is_convertible<T const&, T>::value, "");

    std::tuple t1(T{});
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<T>);

    const T v{};
    std::tuple t2(T{}, 101l, v);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<T, long, T>);
  }
  { // Testing (4)
    int x = 101;
    std::tuple t1(AT, A, 42);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<int>);

    std::tuple t2(AT, A, 42, 0.0, x);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<int, double, int>);
  }
  { // Testing (5)
    using T = ExplicitTestTypes::TestType;
    static_assert(!std::is_convertible<T const&, T>::value, "");

    std::tuple t1(AT, A, T{});
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<T>);

    const T v{};
    std::tuple t2(AT, A, T{}, 101l, v);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<T, long, T>);
  }
  { // Testing (6)
    std::pair<int, char> p1(1, 'c');
    std::tuple t1(AT, A, p1);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<int, char>);

    std::pair<int, std::tuple<char, long, void*>> p2(1, std::tuple<char, long, void*>('c', 3l, nullptr));
    std::tuple t2(AT, A, p2);
    ASSERT_SAME_TYPE(decltype(t2), std::tuple<int, std::tuple<char, long, void*>>);

    int i = 3;
    std::pair<std::reference_wrapper<int>, char> p3(std::ref(i), 'c');
    std::tuple t3(AT, A, p3);
    ASSERT_SAME_TYPE(decltype(t3), std::tuple<std::reference_wrapper<int>, char>);

    std::pair<int&, char> p4(i, 'c');
    std::tuple t4(AT, A, p4);
    ASSERT_SAME_TYPE(decltype(t4), std::tuple<int&, char>);

    std::tuple t5(AT, A, std::pair<int, char>(1, 'c'));
    ASSERT_SAME_TYPE(decltype(t5), std::tuple<int, char>);
  }
  { // Testing (7)
    using Tup = std::tuple<int, decltype(nullptr)>;
    const Tup t(42, nullptr);

    std::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
  { // Testing (8)
    using Tup = std::tuple<void*, unsigned, char>;
    std::tuple t1(Tup(nullptr, 42, 'a'));
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
  { // Testing (9)
    using Tup = std::tuple<int, decltype(nullptr)>;
    const Tup t(42, nullptr);

    std::tuple t1(AT, A, t);
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
  { // Testing (10)
    using Tup = std::tuple<void*, unsigned, char>;
    std::tuple t1(AT, A, Tup(nullptr, 42, 'a'));
    ASSERT_SAME_TYPE(decltype(t1), Tup);
  }
}

// Overloads
//  using A = Allocator
//  using AT = std::allocator_arg_t
// ---------------
// (1)  tuple() -> tuple<>
// (2)  tuple(AT, A const&) -> tuple<>
// (3)  tuple(tuple const&) -> tuple<>
// (4)  tuple(tuple&&) -> tuple<>
// (5)  tuple(AT, A const&, tuple const&) -> tuple<>
// (6)  tuple(AT, A const&, tuple&&) -> tuple<>
void test_empty_specialization()
{
  std::allocator<int> A;
  const auto AT = std::allocator_arg;
  { // Testing (1)
    std::tuple t1{};
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
  { // Testing (2)
    std::tuple t1{AT, A};
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
  { // Testing (3)
    const std::tuple<> t{};
    std::tuple t1(t);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
  { // Testing (4)
    std::tuple t1(std::tuple<>{});
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
  { // Testing (5)
    const std::tuple<> t{};
    std::tuple t1(AT, A, t);
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
  { // Testing (6)
    std::tuple t1(AT, A, std::tuple<>{});
    ASSERT_SAME_TYPE(decltype(t1), std::tuple<>);
  }
}

int main(int, char**) {
  test_primary_template();
  test_empty_specialization();

  return 0;
}
