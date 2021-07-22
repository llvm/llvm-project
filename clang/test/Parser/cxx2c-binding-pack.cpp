// RUN: %clang_cc1 -std=c++2c -verify -fsyntax-only %s

template <unsigned N>
void decompose_array() {
  int arr[4] = {1, 2, 3, 5};
  auto [x, ...rest, ...more_rest] = arr; // expected-error{{multiple ellipses in structured binding declaration}}
}
