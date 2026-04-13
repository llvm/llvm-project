// RUN: %clang_cc1 -emit-llvm-only -verify -std=c++11 %s
struct A {};

enum Foo { F };
typedef Foo Bar; // expected-note{{type 'Bar' (aka 'Foo') found by destructor name lookup}}

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

  a->~foo(); // expected-error{{undeclared identifier 'foo' in destructor name}}

  a->~Bar(); // expected-error{{destructor type 'Bar' (aka 'Foo') in object destruction expression does not match the type 'A' of the object being destroyed}}

  f->~Bar();
  f->~Foo();
  i->~Bar(); // expected-error{{does not match}}

  g().~Bar(); // expected-error{{non-scalar}}

  f->::~Bar(); // expected-error {{not a structure or union}}
  f->::Bar::~Bar();
  f->N::~Wibble(); // expected-error{{'N' does not refer to a type}} expected-error{{'Wibble' does not refer to a type}}

  f->Bar::~Bar(17, 42); // expected-error{{cannot have any arguments}}

  i->~Integer();
  i->Integer::~Integer();
  i->N::~OtherInteger(); // expected-error{{'N' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
                         // expected-error@-1{{'OtherInteger' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->N::OtherInteger::~OtherInteger();
  i->N::OtherInteger::~OtherInteger();
  i->N::OtherInteger::~Integer(); // expected-error{{'Integer' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->N::~Integer(); // expected-error{{'N' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->N::OtherInteger::~Integer(); // expected-error{{'Integer' does not refer to a type name in pseudo-destructor expression; expected the name of type 'int'}}
  i->Integer::~Double(); // expected-error{{the type of object expression ('int') does not match the type being destroyed ('Double' (aka 'double')) in pseudo-destructor expression}}

  ii->~Integer(); // expected-error{{member reference type 'int' is not a pointer; did you mean to use '.'?}}
  ii.~Integer();

  cv_test(a);
  cv_test(f);
  cv_test(i);
  cv_test(d);
}


typedef int Integer;

void destroy_without_call(int *ip) {
  ip->~Integer; // expected-error{{reference to pseudo-destructor must be called}}
}

void paren_destroy_with_call(int *ip) {
  (ip->~Integer)();
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
    p->~oops(); // expected-error{{undeclared identifier 'oops' in destructor name}}
  }

  template void destroy(int*); // expected-note{{in instantiation of function template specialization}}
}

template<typename T> using Id = T;
void AliasTemplate(int *p) {
  p->~Id<int>();
  p->template ~Id<int>(); // expected-error {{'template' keyword not permitted in destructor name}}
  (0).~Id<int>();
  (0).template ~Id<int>(); // expected-error {{'template' keyword not permitted in destructor name}}
}

namespace dotPointerAccess {
struct Base {
  virtual ~Base() {}
};

struct Derived : Base {
  ~Derived() {}
};

void test() {
  Derived d;
  static_cast<Base *>(&d).~Base(); // expected-error {{member reference type 'Base *' is a pointer; did you mean to use '->'}}
  d->~Derived(); // expected-error {{member reference type 'Derived' is not a pointer; did you mean to use '.'}}
}

typedef Derived *Foo;

void test2(Foo d) {
  d.~Foo(); // This is ok
  d.~Derived(); // expected-error {{member reference type 'Foo' (aka 'Derived *') is a pointer; did you mean to use '->'}}
}
}

int pr45294 = 1 .~undeclared_tempate_name<>(); // expected-error {{use of undeclared 'undeclared_tempate_name'}}

namespace TwoPhaseLookup {
  namespace NonTemplate {
    struct Y {};
    using G = Y;
    template<typename T> void f(T *p) { p->~G(); } // expected-error {{no member named '~Y'}}
    void h1(Y *p) { p->~G(); }
    void h2(Y *p) { f(p); }
    namespace N { struct G{}; }
    void h3(N::G *p) { p->~G(); }
    void h4(N::G *p) { f(p); } // expected-note {{instantiation of}}
  }

  namespace NonTemplateUndeclared {
    struct Y {};
    template<typename T> void f(T *p) { p->~G(); } // expected-error {{undeclared identifier 'G' in destructor name}}
    using G = Y;
    void h1(Y *p) { p->~G(); }
    void h2(Y *p) { f(p); } // expected-note {{instantiation of}}
    namespace N { struct G{}; }
    void h3(N::G *p) { p->~G(); }
    void h4(N::G *p) { f(p); }
  }

  namespace Template {
    template<typename T> struct Y {};
    template<class U> using G = Y<U>;
    template<typename T> void f(T *p) { p->~G<int>(); } // expected-error {{no member named '~Y'}}
    void h1(Y<int> *p) { p->~G<int>(); }
    void h2(Y<int> *p) { f(p); }
    namespace N { template<typename T> struct G {}; }
    void h3(N::G<int> *p) { p->~G<int>(); }
    void h4(N::G<int> *p) { f(p); } // expected-note {{instantiation of}}
  }

  namespace TemplateUndeclared {
    template<typename T> struct Y {};
    // FIXME: Formally, this is ill-formed before we hit any instantiation,
    // because we aren't supposed to treat the '<' as introducing a template
    // name.
    template<typename T> void f(T *p) { p->~G<int>(); } // expected-error {{no member named 'G'}}
    template<class U> using G = Y<U>;
    void h1(Y<int> *p) { p->~G<int>(); }
    void h2(Y<int> *p) { f(p); } // expected-note {{instantiation of}}
    namespace N { template<typename T> struct G {}; }
    void h3(N::G<int> *p) { p->~G<int>(); }
    void h4(N::G<int> *p) { f(p); }
  }

  namespace TemplateNamesNonTemplate {
    int A; // expected-note 2{{non-template here}}
    template<typename> int B; // expected-note 2{{variable template 'B' declared here}} expected-warning {{extension}}
    using C = int; // expected-note 2{{non-template here}}

    template<typename T> void f1(int *p) { p->~A<int>(); } // expected-error {{'A' does not refer to a template}}
    template<typename T> void f2(int *p) { p->~B<int>(); } // expected-error {{template name refers to non-type template 'B'}}
    template<typename T> void f3(int *p) { p->~C<int>(); } // expected-error {{'C' does not refer to a template}}
    template<typename T> void f4(int *p) { p->TemplateNamesNonTemplate::C::~A<int>(); } // expected-error {{'A' does not refer to a template}}
    template<typename T> void f5(int *p) { p->TemplateNamesNonTemplate::C::~B<int>(); } // expected-error {{template name refers to non-type template 'TemplateNamesNonTemplate::B'}}
    template<typename T> void f6(int *p) { p->TemplateNamesNonTemplate::C::~C<int>(); } // expected-error {{'C' does not refer to a template}}
  }
}

void destroy_array_element() {
  int arr[5];
  using T = int;
  arr->~T(); // ok, destroy arr[0].
}

void destroy_function() {
  using T = void();
  destroy_function->~T(); // expected-error {{object expression of non-scalar type 'void ()' cannot be used in a pseudo-destructor expression}}
}
