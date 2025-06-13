// RUN: %clang_cc1 -std=c++2c -verify -fsyntax-only %s

template <unsigned N>
void decompose_array() {
  int arr[4] = {1, 2, 3, 5};
  auto [x, ... // #1
    rest, ...more_rest] = arr; // expected-error{{multiple packs in structured binding declaration}}
                               // expected-note@#1{{previous binding pack specified here}}

  auto [y...] = arr; // expected-error{{'...' must immediately precede declared identifier}}

  auto [...] = arr; // #2
                    // expected-error@#2{{expected identifier}}
                    // expected-error@#2{{{no names were provided}}}
                    // expected-warning@#2{{{does not allow a decomposition group to be empty}}}
  auto [a, ..., b] = arr; // #3
                          // expected-error@#3{{expected identifier}}
                          // expected-error@#3{{{only 1 name was provided}}}
  auto [a1, ...] = arr; // #4
                        // expected-error@#4{{expected identifier}}
                        // expected-error@#4{{{only 1 name was provided}}}
  auto [..., b] = arr; // #5
                       // expected-error@#5{{expected identifier}}
                       // expected-error@#5{{{no names were provided}}}
}
