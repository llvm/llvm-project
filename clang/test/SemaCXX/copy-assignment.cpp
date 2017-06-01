// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

#if __cplusplus >= 201103L
// expected-note@+3 2 {{candidate constructor}}
// expected-note@+2 {{passing argument to parameter here}}
#endif
struct A {
};

struct ConvertibleToA {
  operator A();
};

struct ConvertibleToConstA {
#if __cplusplus >= 201103L
// expected-note@+2 {{candidate function}}
#endif
  operator const A();
};

struct B {
  B& operator=(B&);  // expected-note 4 {{candidate function}}
};

struct ConvertibleToB {
  operator B();
};

struct ConvertibleToBref {
  operator B&();
};

struct ConvertibleToConstB {
  operator const B();
};

struct ConvertibleToConstBref {
  operator const B&();
};

struct C {
  int operator=(int); // expected-note{{candidate function}}
  long operator=(long); // expected-note{{candidate function}}
  int operator+=(int); // expected-note{{candidate function}}
  int operator+=(long); // expected-note{{candidate function}}
};

struct D {
  D& operator+=(const D &);
};

struct ConvertibleToInt {
  operator int();
};

void test() {
  A a, na;
  const A constA = A();
  ConvertibleToA convertibleToA;
  ConvertibleToConstA convertibleToConstA;

  B b, nb;
  const B constB = B();
  ConvertibleToB convertibleToB;
  ConvertibleToBref convertibleToBref;
  ConvertibleToConstB convertibleToConstB;
  ConvertibleToConstBref convertibleToConstBref;

  C c, nc;
  const C constC = C();

  D d, nd;
  const D constD = D();

  ConvertibleToInt convertibleToInt;

  na = a;
  na = constA;
  na = convertibleToA;
#if __cplusplus >= 201103L
// expected-error@+2 {{no viable conversion}}
#endif
  na = convertibleToConstA;
  na += a; // expected-error{{no viable overloaded '+='}}

  nb = b;
  nb = constB;  // expected-error{{no viable overloaded '='}}
  nb = convertibleToB; // expected-error{{no viable overloaded '='}}
  nb = convertibleToBref;
  nb = convertibleToConstB; // expected-error{{no viable overloaded '='}}
  nb = convertibleToConstBref; // expected-error{{no viable overloaded '='}}

  nc = c;
  nc = constC;
  nc = 1;
  nc = 1L;
  nc = 1.0; // expected-error{{use of overloaded operator '=' is ambiguous}}
  nc += 1;
  nc += 1L;
  nc += 1.0; // expected-error{{use of overloaded operator '+=' is ambiguous}}

  nd = d;
  nd += d;
  nd += constD;

  int i;
  i = convertibleToInt;
  i = a; // expected-error{{assigning to 'int' from incompatible type 'A'}}
}

// <rdar://problem/8315440>: Don't crash
namespace test1 {
  template<typename T> class A : public unknown::X { // expected-error {{undeclared identifier 'unknown'}} expected-error {{expected class name}}
    A(UndeclaredType n) : X(n) {} // expected-error {{unknown type name 'UndeclaredType'}}
  };
  template<typename T> class B : public A<T>     {
    virtual void foo() {}
  };
  extern template class A<char>;
  extern template class B<char>;
}
