//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <optional>

//   // [optional.optional], class template optional
//   template<class T>
//     class optional;

//     constexpr T* get_ptr() noexcept;
//     constexpr const T* get_ptr() const noexcept;

//   // [optional.optional.ref], partial specialization of optional for lvalue reference types
//   template<class T>
//     class optional<T&>;

//     constexpr T* get_ptr() const noexcept;

#include <cassert>
#include <concepts>
#include <memory>
#include <optional>
#include <utility>

constexpr bool test() {
  // [optional.observe]
  {
    std::optional<int> opt;

    std::same_as<int*> decltype(auto) ptr = opt.get_ptr();
    static_assert(noexcept(opt.get_ptr()));

    assert(ptr == nullptr);

    std::same_as<const int*> decltype(auto) c_ptr = std::as_const(opt).get_ptr();
    static_assert(noexcept(std::as_const(opt).get_ptr()));

    assert(c_ptr == nullptr);
  }
  {
    std::optional<int> opt{94};

    assert(opt.get_ptr() == opt.operator->());

    assert(*opt.get_ptr() == 94);
    assert(opt.get_ptr() == &opt.value());

    assert(std::as_const(opt).get_ptr() == std::as_const(opt).operator->());
    assert(std::as_const(opt).get_ptr() == &std::as_const(opt).value());
    assert(*std::as_const(opt).get_ptr() == 94);
  }
  // [optional.ref.observe]
  {
    std::optional<int&> opt;

    std::same_as<int*> decltype(auto) ptr = opt.get_ptr();
    static_assert(noexcept(opt.get_ptr()));

    assert(ptr == nullptr);

    std::same_as<int*> decltype(auto) c_ptr = std::as_const(opt).get_ptr();
    static_assert(noexcept(std::as_const(opt).get_ptr()));

    assert(c_ptr == nullptr);
  }
  {
    int i = 94;
    std::optional<int&> opt{i};

    assert(opt.get_ptr() == opt.operator->());
    assert(opt.get_ptr() == std::addressof(i));
    assert(*opt.get_ptr() == 94);

    assert(std::as_const(opt).get_ptr() == std::as_const(opt).operator->());
    assert(std::as_const(opt).get_ptr() == std::addressof(i));
    assert(*std::as_const(opt).get_ptr() == 94);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
