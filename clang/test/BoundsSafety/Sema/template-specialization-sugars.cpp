

// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -std=c++17 -verify %s

#include <ptrcheck.h>

// expected-no-diagnostics

struct true_type {
  static constexpr bool value = true;
};

struct false_type {
  static constexpr bool value = false;
};

template <class T>
struct is_ptr : false_type {};

template <class T>
struct is_ptr<T *> : true_type {};

template <class T>
inline constexpr bool is_ptr_v = is_ptr<T>::value;

int main() {
  static_assert(is_ptr_v<int *>);
  static_assert(is_ptr_v<int *__single>);
  static_assert(is_ptr_v<int *__unsafe_indexable>);
  static_assert(is_ptr_v<int *__counted_by(42)>);
  static_assert(is_ptr_v<int *__counted_by_or_null(42)>);
  static_assert(is_ptr_v<int *__sized_by(42)>);
  static_assert(is_ptr_v<int *__sized_by_or_null(42)>);
  static_assert(is_ptr_v<int *__terminated_by(42)>);
  static_assert(is_ptr_v<int *__null_terminated>);

  return 0;
}
