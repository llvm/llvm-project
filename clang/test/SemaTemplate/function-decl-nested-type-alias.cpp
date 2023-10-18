// RUN: %clang_cc1 -x c++ -std=c++14 -fsyntax-only -verify %s

template <class A>
using Type = typename A::NestedType; // expected-error {{type 'float' cannot be used prior to '::' because it has no members}}

template <typename T>
void Func() {
  using MyType = Type<T>(); // expected-note{{in instantiation of template type alias 'Type' requested here}}
  MyType var;
}

void Test() {
  Func<float>(); // expected-note{{in instantiation of function template specialization 'Func<float>' requested here}}
}

