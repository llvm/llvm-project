// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %s

namespace a {
  template <typename T>
  concept C1 = true; // #C1

  template <typename T>
  auto V1 = true; // #V1

  namespace b {
    template <typename T>
    concept C2 = true; // #C2
    template <typename T>
    auto V2 = true; // #V2
  }
}

template <typename T>
concept C3 = true; // #C3
template <typename T>
auto V3 = true; // #V3
template <template <typename T> typename C>
constexpr bool test = true;

static_assert(test<a::C1>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                            // expected-note@#C1 {{here}}
static_assert(test<a::b::C2>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                               // expected-note@#C2 {{here}}
static_assert(test<C3>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                         // expected-note@#C3 {{here}}

static_assert(test<a::V1>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                            // expected-note@#V1 {{here}}
static_assert(test<a::b::V2>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                            // expected-note@#V2 {{here}}
static_assert(test<V3>); // expected-error {{template argument does not refer to a class or alias template, or template template parameter}} \
                         // expected-note@#V3 {{here}}


void f() {
    C3 t1 = 0;  // expected-error {{expected 'auto' or 'decltype(auto)' after concept name}}
    a::C1 t2 = 0; // expected-error {{expected 'auto' or 'decltype(auto)' after concept name}}
    a::b::C2 t3 = 0; // expected-error {{expected 'auto' or 'decltype(auto)' after concept name}}
}
