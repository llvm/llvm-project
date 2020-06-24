// RUN: %clang_cc1 -frecovery-ast -verify %s

bool Foo(int *); // expected-note {{candidate function not viable}} \
                 // expected-note {{candidate function not viable}}

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
