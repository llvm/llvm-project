// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s

void test_const_qualifier() {
  int A = 0;
  int *const __ptr64 B = &A; // expected-note {{variable 'B' declared const here}}
  B += 1; // expected-error {{cannot assign to variable 'B' with const-qualified type 'int *const __ptr64'}}

  int *__ptr32 const C = &A; // expected-note {{variable 'C' declared const here}}
  C = nullptr; // expected-error {{cannot assign to variable 'C' with const-qualified type 'int *const __ptr32'}}
}

void test_type_traits() {
  static_assert(!__is_same(int *, int *const), "");
  static_assert(!__is_same(int *__ptr32, int *__ptr32 const), "");
  static_assert(!__is_same(int *__ptr64, int *__ptr64 const), "");
  static_assert(!__is_same(int *__sptr __ptr32, int *__sptr __ptr32 const), "");
  static_assert(!__is_same(int *__uptr __ptr32, int *__uptr __ptr32 const), "");
  static_assert(!__is_same(int *__ptr32, int *__ptr32 volatile), "");
}
