// RUN: %clang_cc1 -frecovery-ast -verify %s

bool Foo(int *); // expected-note {{candidate function not viable}}

template <typename T>
struct Crash : decltype(Foo(T())) { // expected-error {{no matching function for call to 'Foo'}}
  Crash(){};
};

void test() { Crash<int>(); } // expected-note {{in instantiation of template class}}
