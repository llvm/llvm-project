//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, class... Args>
// concept constructible_from;
//    destructible<T> && is_constructible_v<T, Args...>;

#include <array>
#include <concepts>
#include <memory>
#include <string>
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

template <class T, class... Args>
void test() {
  static_assert(std::constructible_from<T, Args...> ==
                (std::destructible<T> && std::is_constructible_v<T, Args...>));
}

void test() {
  test<bool>();
  test<bool, bool>();

  test<char>();
  test<char, char>();
  test<char, int>();

  test<int>();
  test<int, int>();
  test<int, int, int>();

  test<double, int>();
  test<double, float>();
  test<double, long double>();

  test<void>();
  test<void, bool>();
  test<void, int>();

  test<void*>();
  test<void*, std::nullptr_t>();

  test<int*>();
  test<int*, std::nullptr_t>();
  test<int[], int, int, int>();
  test<int[1]>();
  test<int[1], int>();
  test<int[1], int, int>();

  test<int (*)(int)>();
  test<int (*)(int), int>();
  test<int (*)(int), double>();
  test<int (*)(int), std::nullptr_t>();
  test<int (*)(int), int (*)(int)>();

  test<void (Empty::*)(const int&)>();
  test<void (Empty::*)(const int&), std::nullptr_t>();
  test<void (Empty::*)(const int&) const>();
  test<void (Empty::*)(const int&) const, void (Empty::*)(const int&)>();
  test<void (Empty::*)(const int&) volatile>();
  test<void (Empty::*)(const int&) volatile,
       void (Empty::*)(const int&) const volatile>();
  test<void (Empty::*)(const int&) const volatile>();
  test<void (Empty::*)(const int&) const volatile, double>();
  test<void (Empty::*)(const int&)&>();
  test<void (Empty::*)(const int&)&, void (Empty::*)(const int&) &&>();
  test<void (Empty::*)(const int&) &&>();
  test<void (Empty::*)(const int&)&&, void (Empty::*)(const int&)>();
  test<void (Empty::*)(const int&) throw()>();
  test<void (Empty::*)(const int&) throw(),
       void(Empty::*)(const int&) noexcept(true)>();
  test<void (Empty::*)(const int&) noexcept>();
  test<void (Empty::*)(const int&) noexcept(true)>();
  test<void (Empty::*)(const int&) noexcept(true),
       void (Empty::*)(const int&) noexcept(false)>();
  test<void (Empty::*)(const int&) noexcept(false)>();

  test<int&>();
  test<int&, int>();
  test<int&&>();
  test<int&&, int>();

  test<Empty>();

  test<Defaulted>();
  test<Deleted>();

  test<NoexceptTrue>();
  test<NoexceptFalse>();
  test<Noexcept>();

  test<Protected>();
  test<Private>();

  test<NoexceptDependant<int> >();
  test<NoexceptDependant<double> >();

  test<std::string, char*>();
  test<std::string, const char*>();
  test<std::string, std::string&>();
  test<std::string, std::initializer_list<char> >();

  test<std::unique_ptr<int>, std::unique_ptr<int> >();
  test<std::unique_ptr<int>, std::unique_ptr<int>&>();
  test<std::unique_ptr<int>, std::unique_ptr<int>&&>();

  test<std::array<int, 1> >();
  test<std::array<int, 1>, int>();
  test<std::array<int, 1>, int, int>();
}

int main(int, char**) { return 0; }
