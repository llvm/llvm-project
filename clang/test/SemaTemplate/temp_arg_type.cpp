// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T> class A; // expected-note 2 {{template parameter is declared here}} expected-note{{template is declared here}}

// [temp.arg.type]p1
A<0> *a1; // expected-error{{template argument for template type parameter must be a type}}

A<A> *a2; // expected-error{{use of class template A requires template arguments}}

A<int> *a3;
A<int()> *a4; 
A<int(float)> *a5;
A<A<int> > *a6;

// Pass an overloaded function template:
template<typename T> void function_tpl(T);
A<function_tpl> a7;  // expected-error{{template argument for template type parameter must be a type}}

// Pass a qualified name:
namespace ns {
template<typename T> class B {};  // expected-note{{template is declared here}}
}
A<ns::B> a8; // expected-error{{use of class template ns::B requires template arguments}}

// [temp.arg.type]p2
void f() {
  class X { };
  A<X> * a = 0; // expected-warning{{template argument uses local type 'X'}}
}

struct { int x; } Unnamed; // expected-note{{unnamed type used in template argument was declared here}}
A<__typeof__(Unnamed)> *a9; // expected-warning{{template argument uses unnamed type}}

template<typename T, unsigned N>
struct Array {
  typedef struct { T x[N]; } type;
};

template<typename T> struct A1 { };
A1<Array<int, 17>::type> ax;

// FIXME: [temp.arg.type]p3. The check doesn't really belong here (it
// belongs somewhere in the template instantiation section).
