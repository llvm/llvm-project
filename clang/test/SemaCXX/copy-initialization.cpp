// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class X {
public:
  explicit X(const X&);
  X(int*); // expected-note 3{{candidate constructor}}
  explicit X(float*); // expected-note {{candidate constructor}}
};

class Y : public X { };

void f(Y y, int *ip, float *fp) {
  X x1 = y; // expected-error{{no matching constructor for initialization of 'X'}}
  X x2 = 0;
  X x3 = ip;
  X x4 = fp; // expected-error{{no viable conversion}}
  X x2a(0); // expected-error{{call to constructor of 'X' is ambiguous}}
  X x3a(ip);
  X x4a(fp);
}

struct foo {
 void bar(); // expected-note{{declared here}}
};

// PR3600
void test(const foo *P) { P->bar(); } // expected-error{{'this' argument to member function 'bar' has type 'const foo', but function is not marked const}}

namespace PR6757 {
  struct Foo {
    Foo(); // expected-note{{not viable}}
    Foo(Foo&); // expected-note{{candidate constructor not viable}}
  };

  struct Bar {
    operator const Foo&() const;
  };

  void f(Foo);

  void g(Foo foo) {
    f(Bar()); // expected-error{{no viable constructor copying parameter of type 'const PR6757::Foo'}}
    f(foo);
  }
}

namespace DR5 {
  // Core issue 5: if a temporary is created in copy-initialization, it is of
  // the cv-unqualified version of the destination type.
  namespace Ex1 {
    struct C { };
    C c;
    struct A {
        A(const A&);
        A(const C&);
    };
    const volatile A a = c; // ok
  }

  namespace Ex2 {
    struct S {
      S(S&&);
#if __cplusplus <= 199711L // C++03 or earlier modes
      // expected-warning@-2 {{rvalue references are a C++11 extension}}
#endif
      S(int);
    };
    const S a(0);
    const S b = 0;
  }
}

struct A {};
struct B : A {
  B();
  B(B&);
  B(A);
  B(int);
};
B b = 0; // ok, calls B(int) then A(const A&) then B(A).
