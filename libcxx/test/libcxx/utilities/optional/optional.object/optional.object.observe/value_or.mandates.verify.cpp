//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// Test the mandates (LWG4406 / LWG3424)
// template<class U> constexpr remove_cv_t<T> optional<T>::value_or(U&& v) const &;
// Mandates: is_convertible_v<const T&, remove_cv_t<T>> is true
//           and is_convertible_v<U, remove_cv_t<T>> is true.
//
// template<class U> constexpr remove_cv_t<T> optional<T>::value_or(U&& v) &&;
// Mandates: is_convertible_v<T, remove_cv_t<T>> is true
//           and is_convertible_v<U, remove_cv_t<T>> is true.
//
// template<class U> constexpr remove_cv_t<T> optional<T&>::value_or(U&& v) const; // since C++26
// Mandates: is_convertible_v<T&, remove_cv_t<T>> is true
//           and is_convertible_v<U, remove_cv_t<T>> is true.

#include <optional>
#include <utility>

#include "test_macros.h"

struct NonCopyable {
  NonCopyable(int) {}
  NonCopyable(const NonCopyable&) = delete;
};

struct NonMovable {
  NonMovable(int) {}
  NonMovable(NonMovable&&) = delete;
};

struct NotConvertibleFromInt {};

void test() {
  // const& overload
  // !is_convertible_v<const T&, remove_cv_t<T>>
  {
    const std::optional<NonCopyable> o{5};
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NonCopyable>::value_or<int>' requested here}}
    (void)o.value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}const T& must be implicitly convertible to remove_cv_t<T>}}
  }

  // const& overload
  // !is_convertible_v<U, remove_cv_t<T>>
  {
    const std::optional<NotConvertibleFromInt> o{std::in_place};
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NotConvertibleFromInt>::value_or<int>' requested here}}
    (void)o.value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}optional<T>::value_or: U must be implicitly convertible to remove_cv_t<T>}}
  }

  // && overload
  // !is_convertible_v<T, remove_cv_t<T>>
  {
    std::optional<NonMovable> o{5};
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NonMovable>::value_or<int>' requested here}}
    (void)std::move(o).value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}optional<T>::value_or: T must be implicitly convertible to remove_cv_t<T>}}
  }

  // && overload
  // !is_convertible_v<U, remove_cv_t<T>>
  {
    std::optional<NotConvertibleFromInt> o{std::in_place};
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NotConvertibleFromInt>::value_or<int>' requested here}}
    (void)std::move(o).value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}optional<T>::value_or: U must be implicitly convertible to remove_cv_t<T>}}
  }

#if TEST_STD_VER >= 26
  // optional<T&>::value_or
  // !is_convertible_v<T&, remove_cv_t<T>>
  {
    NonCopyable nc{5};
    std::optional<NonCopyable&> o(nc);
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NonCopyable &>::value_or<int>' requested here}}
    (void)o.value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}T& must be implicitly convertible to remove_cv_t<T>}}
  }

  // optional<T&>::value_or
  // !is_convertible_v<U, remove_cv_t<T>>
  {
    NotConvertibleFromInt nci;
    std::optional<NotConvertibleFromInt&> o(nci);
    // expected-note@+1 {{in instantiation of function template specialization 'std::optional<NotConvertibleFromInt &>::value_or<int>' requested here}}
    (void)o.value_or(5);
    // expected-error-re@*:* {{static assertion failed {{.*}}optional<T&>::value_or: U must be implicitly convertible to remove_cv_t<T>}}
  }
#endif
}
