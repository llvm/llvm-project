// RUN: %clang_cc1 -std=c++20 -verify -emit-llvm-only %s
// Regression test for https://github.com/llvm/llvm-project/issues/181410
//
// A class template partial specialization diagnosed as "not more specialized
// than the primary template" was not marked invalid, so it was still selected
// during template argument deduction. Using it for instantiation produced
// dependent expressions in non-dependent contexts, causing an assertion
// failure in CodeGen: assert(!Init->isValueDependent()).

template <int>
struct integer_sequence {};

template <int>
struct array {};

template <int*>
struct MetaValuesHelper; // expected-note 2{{template is declared here}}

template <typename TupleName, TupleName kValues>
struct MetaValuesHelper<kValues> { // expected-error {{class template partial specialization is not more specialized than the primary template}}
  template <int... Is>
  static array<stdget<Is>(kValues)...> MetaValuesFunc(integer_sequence<Is...>);
};

int kBaseIndexRegistersUsed;

// Previously this crashed with: assert(!Init->isValueDependent())
// Now the invalid partial specialization is excluded from deduction,
// and the primary template (which is only forward-declared) is used instead.
array<0> u = decltype(MetaValuesHelper<&kBaseIndexRegistersUsed>::MetaValuesFunc(integer_sequence<0>{})){}; // expected-error {{implicit instantiation of undefined template 'MetaValuesHelper<&kBaseIndexRegistersUsed>'}}
