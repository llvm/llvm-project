// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A {};

enum Foo { F };
typedef Foo Bar; // expected-note{{type 'Bar' (aka 'Foo') is declared here}}

typedef int Integer;
typedef double Double;

void g();

namespace N {
  typedef Foo Wibble;
  typedef int OtherInteger;
}

template <typename T>
void cv_test(const volatile T* cvt) {
  cvt->T::~T(); // no-warning
}

void f(A* a, Foo *f, int *i, double *d, int ii) {
  a->~A();
  a->A::~A();
  
  a->~foo(); // expected-error{{identifier 'foo' in object destruction expression does not name a type}}
  
  a->~Bar(); // expected-error{{destructor type 'Bar' (aka 'Foo') in object destruction expression does not match the type 'A' of the object being destroyed}}
  
  f->~Bar();
  f->~Foo();
  i->~Bar(); // expected-error{{does not match}}
  
  g().~Bar(); // expected-error{{non-scalar}}
  
  f->::~Bar();
  f->N::~Wibble(); // FIXME: technically, Wibble isn't a class-name
  
  f->::~Bar(17, 42); // expected-error{{cannot have any arguments}}

  i->~Integer();
  i->Integer::~Integer();
  i->N::~OtherInteger();
  i->N::OtherInteger::~OtherInteger();
  i->N::OtherInteger::~Integer(); // expected-error{{'Integer' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->N::~Integer(); // expected-error{{'Integer' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->Integer::~Double(); // expected-error{{the type of object expression ('int') does not match the type being destroyed ('Double' (aka 'double')) in pseudo-destructor expression}}

  ii->~Integer(); // expected-error{{member reference type 'int' is not a pointer; maybe you meant to use '.'?}}
  ii.~Integer();

  cv_test(a);
  cv_test(f);
  cv_test(i);
  cv_test(d);
}


typedef int Integer;

void destroy_without_call(int *ip) {
  ip->~Integer; // expected-error{{called immediately}}
}

// PR5530
namespace N1 {
  class X0 { };
}

void test_X0(N1::X0 &x0) {
  x0.~X0();
}

namespace PR11339 {
  template<class T>
  void destroy(T* p) {
    p->~T(); // ok
    p->~oops(); // expected-error{{expected the class name after '~' to name a destructor}}
  }

  template void destroy(int*); // expected-note{{in instantiation of function template specialization}}
}
