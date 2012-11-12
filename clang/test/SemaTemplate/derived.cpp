// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> class vector2 {};
template<typename T> class vector : vector2<T> {};

template<typename T> void Foo2(vector2<const T*> V) {}  // expected-note{{candidate template ignored: can't deduce a type for 'T' which would make 'const T' equal 'int'}}
template<typename T> void Foo(vector<const T*> V) {} // expected-note {{candidate template ignored: can't deduce a type for 'T' which would make 'const T' equal 'int'}}

void test() {
  Foo2(vector2<int*>());  // expected-error{{no matching function for call to 'Foo2'}}
  Foo(vector<int*>());  // expected-error{{no matching function for call to 'Foo'}}
}
