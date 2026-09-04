// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 %s

namespace invalid_partial_spec {
  template <int> struct integer_sequence {};
  template <int> struct array {};
  template <int*> struct MetaValuesHelper; // expected-note {{template is declared here}}

  template <typename TupleName, TupleName kValues>
  struct MetaValuesHelper<kValues> {
    // expected-error@-1 {{class template partial specialization is not more specialized than the primary template}}
    template <int... Is>
    static array<undefined<Is>(kValues)...> MetaValuesFunc(integer_sequence<Is...>);
    // expected-note@-1 {{candidate template ignored: substitution failure [with Is = <0>]: use of undeclared identifier 'undefined'}}
  };

  int kBaseIndexRegistersUsed;

  constexpr array<0> GenMachineInsnInfos() {
    return decltype(MetaValuesHelper<&kBaseIndexRegistersUsed>::MetaValuesFunc(integer_sequence<0>{})){};
    // expected-error@-1 {{no matching function for call to 'MetaValuesFunc'}}
  }

  array<0> u = GenMachineInsnInfos();
} // namspace invalid_partial_spec
