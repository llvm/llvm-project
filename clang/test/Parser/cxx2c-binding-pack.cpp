// RUN: %clang_cc1 -std=c++2c -verify -fsyntax-only %s

template <unsigned N>
void decompose_array() {
  int arr[4] = {1, 2, 3, 5};
  auto [x, ... // #1
    rest, ...more_rest] = arr; // expected-error{{multiple ellipses in structured binding declaration}}
                               // expected-note@#1{{previous ellipsis specified here}}
                               //
  auto [y...] = arr; // expected-error{{'...' must immediately precede declared identifier}}
}
