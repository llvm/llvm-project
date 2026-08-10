// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A { 
  struct B { void f(); }; 
  int a; 
  int Y;
};

template<class B, class a> struct X : A { 
  B b;  // A's B 
  a c;  // expected-error{{unknown type name 'a'}} 

  void g() {
    b.g(); // expected-error{{no member named 'g' in 'A::B'}}
  }
};
