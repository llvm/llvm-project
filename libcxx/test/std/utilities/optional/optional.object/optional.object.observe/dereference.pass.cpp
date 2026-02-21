//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T& optional<T>::operator*() &;
// constexpr const T& optional<T>::operator*() const&;
// constexpr T&& optional<T>::operator*() &&;
// constexpr const T&& optional<T>::operator*() const&&;
// constexpr T& optional<T&>::operator*() const;

#include <cassert>
#include <memory>
#include <optional>
#include <utility>

#include "test_macros.h"
#if TEST_STD_VER >= 26
#  include "copy_move_types.h"
#endif

struct X {
  constexpr int test() const& { return 3; }
  constexpr int test() & { return 4; }
  constexpr int test() const&& { return 5; }
  constexpr int test() && { return 6; }
};

template <typename T>
constexpr void test_contract() {
  std::optional<T> opt(T{});

  ASSERT_SAME_TYPE(decltype(*opt), T&);
  ASSERT_SAME_TYPE(decltype(*std::move(opt)), T&&);
  ASSERT_SAME_TYPE(decltype(*std::as_const(opt)), const T&);
  ASSERT_SAME_TYPE(decltype(*std::move(std::as_const(opt))), const T&&);

  ASSERT_NOEXCEPT(*opt);
  ASSERT_NOEXCEPT(*std::move(opt));
  ASSERT_NOEXCEPT(*std::as_const(opt));
  ASSERT_NOEXCEPT(*std::move(std::as_const(opt)));
}

#if TEST_STD_VER >= 26
template <typename T>
constexpr void test_ref_contract() {
  std::optional<T&> opt;
  ASSERT_SAME_TYPE(decltype(*opt), T&);
  ASSERT_SAME_TYPE(decltype(*std::move(opt)), T&);
  ASSERT_SAME_TYPE(decltype(*std::as_const(opt)), T&);
  ASSERT_SAME_TYPE(decltype(*std::move(std::as_const(opt))), T&);

  ASSERT_NOEXCEPT(*opt);
  ASSERT_NOEXCEPT(*std::move(opt));
  ASSERT_NOEXCEPT(*std::as_const(opt));
  ASSERT_NOEXCEPT(*std::move(std::as_const(opt)));
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

  {
    std::optional<X> opt{X{}};
    assert((*std::as_const(opt)).test() == 3);
    assert((*opt).test() == 4);
    assert((*std::move(std::as_const(opt))).test() == 5);
    assert((*std::move(opt)).test() == 6);

    // Test that operator* returns a stable reference
    X& x = *opt;
    assert(&x == &(*opt));

    const X& x2 = *std::as_const(opt);
    assert(&x2 == &(*opt));
  }

#if TEST_STD_VER >= 26
  test_ref_contract<int>();
  test_ref_contract<float>();
  test_ref_contract<double>();
  test_ref_contract<X>();
  test_ref_contract<const int>();
  test_ref_contract<const float>();
  test_ref_contract<const double>();
  test_ref_contract<const X>();

  {
    TracedCopyMove x{};
    std::optional<TracedCopyMove&> o1(x);

    assert(std::addressof(*o1) == std::addressof(x));
    assert(std::addressof(*std::move(o1)) == std::addressof(x));
    assert(std::addressof(*std::as_const(o1)) == std::addressof(x));
    assert(std::addressof(*std::move(std::as_const(o1))) == std::addressof(x));
    assert(x.constMove == 0);
    assert(x.nonConstMove == 0);
    assert(x.constCopy == 0);
    assert(x.nonConstCopy == 0);
  }
  {
    TracedCopyMove x{};
    std::optional<const TracedCopyMove&> o2(x);

    assert(std::addressof(*o2) == std::addressof(x));
    assert(std::addressof(*std::move(o2)) == std::addressof(x));
    assert(std::addressof(*std::as_const(o2)) == std::addressof(x));
    assert(std::addressof(*std::move(std::as_const(o2))) == std::addressof(x));
    assert(x.constMove == 0);
    assert(x.nonConstMove == 0);
    assert(x.constCopy == 0);
    assert(x.nonConstCopy == 0);
  }
  {
    // Verify that optional<T&>::operator* always returns a T&
    X x{};
    std::optional<X&> o3(x);
    assert((*o3).test() == 4);
    assert((*std::as_const(o3)).test() == 4);
    assert((*std::move(o3)).test() == 4);
    assert((*std::move(std::as_const(o3))).test() == 4);
  }
#endif

  return true;
}

int main(int, char**) {
  assert(test());
  static_assert(test());

  return 0;
}
