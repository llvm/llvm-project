// RUN: %clang_cc1 -frecovery-ast -verify %s

bool Foo(int *); // expected-note 3{{candidate function not viable}}

template <typename T>
struct Crash : decltype(Foo(T())) { // expected-error {{no matching function for call to 'Foo'}}
  Crash(){};
};

void test() { Crash<int>(); } // expected-note {{in instantiation of template class}}

template <typename T>
using Alias = decltype(Foo(T())); // expected-error {{no matching function for call to 'Foo'}}
template <typename T>
struct Crash2 : decltype(Alias<T>()) { // expected-note {{in instantiation of template type alias 'Alias' requested here}}
  Crash2(){};
};

void test2() { Crash2<int>(); } // expected-note {{in instantiation of template class 'Crash2<int>' requested here}}

template <typename T>
class Base {};
template <typename T>
struct Crash3 : Base<decltype(Foo(T()))> { // expected-error {{no matching function for call to 'Foo'}}
  Crash3(){};
};

void test3() { Crash3<int>(); } // expected-note {{in instantiation of template class}}
