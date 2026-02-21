// RUN: %clang_cc1 -std=c++20 -verify -emit-llvm-only %s
// https://github.com/llvm/llvm-project/issues/181410

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

array<0> u = decltype(MetaValuesHelper<&kBaseIndexRegistersUsed>::MetaValuesFunc(integer_sequence<0>{})){}; // expected-error {{implicit instantiation of undefined template 'MetaValuesHelper<&kBaseIndexRegistersUsed>'}}
