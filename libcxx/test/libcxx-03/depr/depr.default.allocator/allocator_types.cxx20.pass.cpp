//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Check that the following types are provided regardless of the Standard when
// we request them from libc++.

// template <class T>
// class allocator
// {
// public:
//     typedef size_t                                size_type;
//     typedef ptrdiff_t                             difference_type;
//     typedef T*                                    pointer;
//     typedef const T*                              const_pointer;
//     typedef typename add_lvalue_reference<T>::type       reference;
//     typedef typename add_lvalue_reference<const T>::type const_reference;
//
//     template <class U> struct rebind {typedef allocator<U> other;};
// ...
// };

// Removed in C++20, deprecated in C++17.

// REQUIRES: c++03 || c++11 || c++14 || c++17
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <type_traits>
#include <cstddef>

template <class T>
void test() {
  static_assert((std::is_same<typename std::allocator<T>::size_type, std::size_t>::value), "");
  static_assert((std::is_same<typename std::allocator<T>::difference_type, std::ptrdiff_t>::value), "");
  static_assert((std::is_same<typename std::allocator<T>::pointer, T*>::value), "");
  static_assert((std::is_same<typename std::allocator<T>::const_pointer, const T*>::value), "");
  static_assert((std::is_same<typename std::allocator<T>::reference, T&>::value), "");
  static_assert((std::is_same<typename std::allocator<T>::const_reference, const T&>::value), "");
  static_assert(
      (std::is_same<typename std::allocator<T>::template rebind<int>::other, std::allocator<int> >::value), "");
}

int main(int, char**) {
  test<char>();
  return 0;
}
