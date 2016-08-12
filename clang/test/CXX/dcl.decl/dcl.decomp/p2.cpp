// RUN: %clang_cc1 -std=c++1z -verify %s

int array() {
  static int arr[3] = {};
  // FIXME: We are supposed to create an array object here and perform elementwise initialization.
  auto [a, b, c] = arr; // expected-error {{cannot decompose non-class, non-array}}

  auto &[d, e] = arr; // expected-error {{type 'int [3]' decomposes into 3 elements, but only 2 names were provided}}
  auto &[f, g, h, i] = arr; // expected-error {{type 'int [3]' decomposes into 3 elements, but 4 names were provided}}

  auto &[r0, r1, r2] = arr;
  const auto &[cr0, cr1, cr2] = arr;

  static_assert(&arr[0] == &r0);
  static_assert(&arr[0] == &cr0);

  using T = int;
  using T = decltype(r0);
  using U = const int;
  using U = decltype(cr0);

  return r1 + cr2;
}
