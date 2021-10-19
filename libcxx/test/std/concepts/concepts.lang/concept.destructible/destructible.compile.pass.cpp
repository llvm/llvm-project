//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// template<class T>
// concept destructible = is_nothrow_destructible_v<T>;

#include <concepts>
#include <type_traits>

struct Empty {};

struct Defaulted {
  ~Defaulted() = default;
};
struct Deleted {
  ~Deleted() = delete;
};

struct Noexcept {
  ~Noexcept() noexcept;
};
struct NoexceptTrue {
  ~NoexceptTrue() noexcept(true);
};
struct NoexceptFalse {
  ~NoexceptFalse() noexcept(false);
};

struct Protected {
protected:
  ~Protected() = default;
};
struct Private {
private:
  ~Private() = default;
};

template <class T>
struct NoexceptDependant {
  ~NoexceptDependant() noexcept(std::is_same_v<T, int>);
};

template <class T>
void test() {
  static_assert(std::destructible<T> == std::is_nothrow_destructible_v<T>);
}

void test() {
  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<Noexcept>();
  test<NoexceptTrue>();
  test<NoexceptFalse>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int> >();
  test<NoexceptDependant<double> >();

  test<bool>();
  test<char>();
  test<int>();
  test<double>();
}

// Required for MSVC internal test runner compatibility.
int main(int, char**) { return 0; }
