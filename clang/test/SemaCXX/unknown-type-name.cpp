// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR3990
namespace N {
  struct Wibble {
  };

  typedef Wibble foo;

  int zeppelin; // expected-note{{declared here}}
}
using namespace N;

foo::bar x; // expected-error{{no type named 'bar' in 'N::Wibble'}}

void f() {
  foo::bar  = 4; // expected-error{{no member named 'bar' in 'N::Wibble'}}
}

int f(foo::bar); // expected-error{{no type named 'bar' in 'N::Wibble'}}

int f(doulbe); // expected-error{{did you mean 'double'?}}

int fun(zapotron); // expected-error{{unknown type name 'zapotron'}}
int var(zepelin); // expected-error{{did you mean 'zeppelin'?}}

template<typename T>
struct A {
  typedef T type;

  type f();

  type g();

  static int n;
  static type m;
  static int h(T::type, int); // expected-error{{missing 'typename'}}
  static int h(T::type x, char); // expected-error{{missing 'typename'}}
};

template<typename T>
A<T>::type g(T t) { return t; } // expected-error{{missing 'typename'}}

template<typename T>
A<T>::type A<T>::f() { return type(); } // expected-error{{missing 'typename'}}

template<typename T>
void f(T::type) { } // expected-error{{missing 'typename'}}

template<typename T>
void g(T::type x) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(T::type, int) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(T::type x, char) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(int, T::type) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(char, T::type x) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(int, T::type, int) { } // expected-error{{missing 'typename'}}

template<typename T>
void f(int, T::type x, char) { } // expected-error{{missing 'typename'}}

int *p;

// FIXME: We should assume that 'undeclared' is a type, not a parameter name
//        here, and produce an 'unknown type name' diagnostic instead.
int f1(undeclared, int); // expected-error{{requires a type specifier}}

int f2(undeclared, 0); // expected-error{{undeclared identifier}}

int f3(undeclared *p, int); // expected-error{{unknown type name 'undeclared'}}

int f4(undeclared *p, 0); // expected-error{{undeclared identifier}}

int *test(UnknownType *fool) { return 0; } // expected-error{{unknown type name 'UnknownType'}}

template<typename T> int A<T>::n(T::value); // ok
template<typename T>
A<T>::type // expected-error{{missing 'typename'}}
A<T>::m(T::value, 0); // ok

template<typename T> int A<T>::h(T::type, int) {} // expected-error{{missing 'typename'}}
template<typename T> int A<T>::h(T::type x, char) {} // expected-error{{missing 'typename'}}

template<typename T> int h(T::type, int); // expected-error{{missing 'typename'}}
template<typename T> int h(T::type x, char); // expected-error{{missing 'typename'}}

template<typename T> int junk1(T::junk); // expected-warning{{variable templates are a C++14 extension}}
template<typename T> int junk2(T::junk) throw(); // expected-error{{missing 'typename'}}
template<typename T> int junk3(T::junk) = delete; // expected-error{{missing 'typename'}} expected-warning{{C++11}}
template<typename T> int junk4(T::junk j); // expected-error{{missing 'typename'}}

// FIXME: We can tell this was intended to be a function because it does not
//        have a dependent nested name specifier.
template<typename T> int i(T::type, int()); // expected-warning{{variable templates are a C++14 extension}}

// FIXME: We know which type specifier should have been specified here. Provide
//        a fix-it to add 'typename A<T>::type'
template<typename T>
A<T>::g() { } // expected-error{{requires a type specifier}}
