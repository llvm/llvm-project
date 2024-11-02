//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that all the type_trait aliases can be mangled

// UNSUPPORTED: c++03, c++11

// ignore deprecated volatile return types
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-deprecated-volatile
// MSVC warning C5216: 'volatile int' a volatile qualified return type is deprecated in C++20
// ADDITIONAL_COMPILE_FLAGS(cl-style-warnings): /wd5216

#include <type_traits>
#include <utility>

#include "test_macros.h"

template <class T>
std::add_const_t<T> add_const() {
  return {};
}

template <class T>
std::add_cv_t<T> add_cv() {
  return {};
}

template <class T>
std::add_lvalue_reference_t<T> add_lvalue_reference() {
  static int i;
  return i;
}

template <class T>
std::add_pointer_t<T> add_pointer() {
  return {};
}

template <class T>
std::add_rvalue_reference_t<T> add_rvalue_reference() {
  static int i;
  return std::move(i);
}

template <class T>
std::add_volatile_t<T> add_volatile() {
  static int i;
  return std::move(i);
}

template <class T>
std::conditional_t<true, T, T> conditional() {
  return {};
}

template <class T>
std::decay_t<T> decay() {
  return {};
}

template <class T>
std::enable_if_t<true, T> enable_if() {
  return {};
}

template <class T>
std::make_signed_t<T> make_signed() {
  return {};
}

template <class T>
std::make_unsigned_t<T> make_unsigned() {
  return {};
}

template <class T>
std::remove_all_extents_t<T> remove_all_extents() {
  return {};
}

template <class T>
std::remove_const_t<T> remove_const() {
  return {};
}

template <class T>
std::remove_cv_t<T> remove_cv() {
  return {};
}

#if TEST_STD_VER >= 20
template <class T>
std::remove_cvref_t<T> remove_cvref() {
  return {};
}
#endif

template <class T>
std::remove_extent_t<T> remove_extent() {
  return {};
}

template <class T>
std::remove_pointer_t<T> remove_pointer() {
  return {};
}

template <class T>
std::remove_reference_t<T> remove_reference() {
  return {};
}

template <class T>
std::remove_volatile_t<T> remove_volatile() {
  return {};
}

#if TEST_STD_VER >= 20
template <class T>
std::type_identity_t<T> type_identity() {
  return {};
}
#endif

template <class T>
std::underlying_type_t<T> underlying_type() {
  return {};
}

enum class E : int {};

#if TEST_STD_VER >= 17
template <class T>
std::void_t<T> void_t() {}
#endif

void instantiate() {
  add_const<int>();
  add_cv<int>();
  add_lvalue_reference<int>();
  add_pointer<int>();
  add_rvalue_reference<int>();
  add_volatile<int>();
  decay<int>();
  enable_if<int>();
  make_signed<int>();
  make_unsigned<int>();
  remove_all_extents<int>();
  remove_const<int>();
  remove_cv<int>();
#if TEST_STD_VER >= 20
  remove_cvref<int>();
#endif
  remove_extent<int>();
  remove_pointer<int>();
  remove_reference<int>();
  remove_volatile<int>();
#if TEST_STD_VER >= 20
  type_identity<int>();
#endif
  underlying_type<E>();
#if TEST_STD_VER >= 17
  void_t<int>();
#endif
}
