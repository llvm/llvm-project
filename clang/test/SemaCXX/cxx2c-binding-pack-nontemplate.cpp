// RUN: %clang_cc1 -fsyntax-only -std=c++2c %s -verify

void decompose_array() {
  int arr[4] = {1, 2, 3, 6};
  auto [x, ...rest, y] = arr; // expected-error{{pack declaration outside of template}}
}

template <unsigned N>
void decompose_array_2() {
  int arr[4] = {1, 2, 3, N};
  auto [x, ...rest, ...y] = arr; // expected-error{{multiple ellipses in structured binding declaration}}
}
