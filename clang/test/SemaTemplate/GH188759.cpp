// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 %s

namespace t1 {
  template <int> struct integer_sequence {};
  template <int> struct array {};
  template <int ARRAY_SIZE, array<ARRAY_SIZE> test_apdus> void runBlobs() {
    []<int... INDEX>(integer_sequence<INDEX...>) { // expected-note {{requested here}}
      int x{operator0<test_apdus, INDEX>()...};
      // expected-error@-1 {{use of undeclared identifier 'operator0'}}
    }(integer_sequence<1>{});
  }
  template void runBlobs<2, {}>(); // expected-note {{requested here}}
} // namespace t1
