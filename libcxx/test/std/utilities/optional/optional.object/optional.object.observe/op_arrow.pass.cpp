//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T* optional<T>::operator->() noexcept;
// constexpr const T* optional<T>::operator->() const noexcept;
// constexpr T* optional<T&>::operator->() const noexcept;

#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "test_macros.h"

struct X {
  constexpr int test() noexcept { return 3; }
  constexpr int test() const noexcept { return 4; }
};

template <typename T>
constexpr void test_contract() {
  std::optional<T> opt;
  ASSERT_SAME_TYPE(decltype(opt.operator->()), T*);
  ASSERT_SAME_TYPE(decltype(std::as_const(opt).operator->()), const T*);

  ASSERT_NOEXCEPT(opt.operator->());
  ASSERT_NOEXCEPT(std::as_const(opt).operator->());
}

#if TEST_STD_VER >= 26
template <typename T>
constexpr void test_ref_contract() {
  std::optional<T&> opt;

  ASSERT_SAME_TYPE(decltype(opt.operator->()), T*);
  ASSERT_SAME_TYPE(decltype(std::as_const(opt).operator->()), T*);

  ASSERT_NOEXCEPT(opt.operator->());
  ASSERT_NOEXCEPT(std::as_const(opt).operator->());
}

#endif

constexpr bool test() {
  test_contract<int>();
  test_contract<float>();
  test_contract<double>();
  test_contract<X>();

  test_contract<const int>();
  test_contract<const float>();
  test_contract<const double>();
  test_contract<const X>();

  std::optional<X> opt{X{}};
  {
    assert(opt->test() == 3);
    assert(std::as_const(opt)->test() == 4);
    assert(opt.operator->() == &*opt);
    assert(std::as_const(opt).operator->() == &*opt);
  }

#if TEST_STD_VER >= 26
  test_ref_contract<int>();
  test_ref_contract<float>();
  test_ref_contract<double>();
  test_ref_contract<X>();

  test_ref_contract<const int>();
  test_ref_contract<const float>();
  test_ref_contract<const double>();

  {
    X x{};
    std::optional<X&> o1(x);
    ASSERT_SAME_TYPE(decltype(o1.operator->()), X*);
    ASSERT_NOEXCEPT(o1.operator->());
    assert(o1.operator->() == std::addressof(x));
    assert(o1->test() == 3);
  }
  {
    X x{};
    std::optional<const X&> o2(x);
    ASSERT_SAME_TYPE(decltype(o2.operator->()), const X*);
    ASSERT_NOEXCEPT(o2.operator->());
    assert(o2.operator->() == std::addressof(x));
    assert(o2->test() == 4);
  }
  {
    X x{};
    const std::optional<X&> o3(x);
    ASSERT_SAME_TYPE(decltype(o3.operator->()), X*);
    ASSERT_NOEXCEPT(o3.operator->());
    assert(o3.operator->() == std::addressof(x));
    assert(o3->test() == 3);
  }
  {
    X x{};
    const std::optional<const X&> o4(x);
    ASSERT_SAME_TYPE(decltype(o4.operator->()), const X*);
    ASSERT_NOEXCEPT(o4.operator->());
    assert(o4.operator->() == std::addressof(x));
    assert(o4->test() == 4);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
