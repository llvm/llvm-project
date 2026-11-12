//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;
//     Should not require a specialization of pointer_traits for Ptr.

#include <memory>
#include <type_traits>
#include <utility>

struct IntPtr {
  constexpr int* operator->() const { return ptr; }

  int* ptr;
};

template <class T, bool>
struct TemplatedPtr {
  constexpr T* operator->() const { return ptr; }

  T* ptr;
};

template <template <class...> class Templ, class Ignore, class... Args>
struct is_valid_expansion_impl : std::false_type {};

template <template <class...> class Templ, class... Args>
struct is_valid_expansion_impl<Templ, decltype((void)Templ<Args...>{}, 0), Args...> : std::true_type {};

template <template <class...> class Templ, class... Args>
using is_valid_expansion = is_valid_expansion_impl<Templ, int, Args...>;

template <class Ptr>
using TestToAddressCall = decltype(std::to_address(std::declval<Ptr>()));

constexpr bool test() {
  int i = 0;

  static_assert(std::to_address(IntPtr{nullptr}) == nullptr);
  static_assert(std::to_address(IntPtr{&i}) == &i);

  bool b = false;

  static_assert(std::to_address(TemplatedPtr<bool, true>{nullptr}) == nullptr);
  static_assert(std::to_address(TemplatedPtr<bool, true>{&b}) == &b);

  static_assert(!is_valid_expansion<TestToAddressCall, int>::value);
  static_assert(is_valid_expansion<TestToAddressCall, IntPtr>::value);
  static_assert(is_valid_expansion<TestToAddressCall, TemplatedPtr<bool, true>>::value);

  return true;
}

int main(int, char**) {
  static_assert(test());
  return 0;
}
