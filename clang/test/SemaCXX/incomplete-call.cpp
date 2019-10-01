// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A; // expected-note 14 {{forward declaration of 'A'}}

A f(); // expected-note {{'f' declared here}}

struct B {
  A f(); // expected-note {{'f' declared here}}
  A operator()(); // expected-note 2 {{'operator()' declared here}}
  operator A(); // expected-note {{'operator A' declared here}}
  A operator!(); // expected-note 2 {{'operator!' declared here}}
  A operator++(int); // expected-note {{'operator++' declared here}}
  A operator[](int); // expected-note {{'operator[]' declared here}}
  A operator+(int); // expected-note {{'operator+' declared here}}
  A operator->(); // expected-note {{'operator->' declared here}}
};

void g() {
  f(); // expected-error {{calling 'f' with incomplete return type 'A'}}

  typedef A (*Func)();
  Func fp;
  fp(); // expected-error {{calling function with incomplete return type 'A'}}
  ((Func)0)();  // expected-error {{calling function with incomplete return type 'A'}}  
  
  B b;
  b.f(); // expected-error {{calling 'f' with incomplete return type 'A'}}
  
  b.operator()(); // expected-error {{calling 'operator()' with incomplete return type 'A'}}
  b.operator A(); // expected-error {{calling 'operator A' with incomplete return type 'A'}}
  b.operator!(); // expected-error {{calling 'operator!' with incomplete return type 'A'}}
  
  !b; // expected-error {{calling 'operator!' with incomplete return type 'A'}}
  b(); // expected-error {{calling 'operator()' with incomplete return type 'A'}}
  b++; // expected-error {{calling 'operator++' with incomplete return type 'A'}}
  b[0]; // expected-error {{calling 'operator[]' with incomplete return type 'A'}}
  b + 1; // expected-error {{calling 'operator+' with incomplete return type 'A'}}
  b->f(); // expected-error {{calling 'operator->' with incomplete return type 'A'}}
  
  A (B::*mfp)() = 0;
  (b.*mfp)(); // expected-error {{calling function with incomplete return type 'A'}}
  
}


struct C; // expected-note{{forward declaration}}

void test_incomplete_object_call(C& c) {
  c(); // expected-error{{incomplete type in call to object of type}}
}

void test_incomplete_object_dtor(C *p) {
  p.~C(); // expected-error{{member reference type 'C *' is a pointer; did you mean to use '->'?}}
}

namespace pr18542 {
  struct X {
    int count;
    template<typename CharT> class basic_istream;
    template<typename CharT>
      void basic_istream<CharT>::read() { // expected-error{{out-of-line definition of 'read' from class 'basic_istream<CharT>' without definition}}
        count = 0;
      }
  };
}

