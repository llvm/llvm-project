// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// rdar4641403
namespace N {
  struct X { // expected-note{{candidate found by name lookup}}
    float b;
  };
}

using namespace N;

typedef struct {
  int a;
} X; // expected-note{{candidate found by name lookup}}


struct Y { };
void Y(int) { }

void f() {
  X *x; // expected-error{{reference to 'X' is ambiguous}}
  Y(1); // okay
}

namespace PR17731 {
  void f() {
    struct S { S() {} };
    int S(void);
    int a = S();
    struct S b;
    {
      int S(void);
      int a = S();
      struct S c = b;
    }
    {
      struct S { S() {} }; // expected-note {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
      // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif
      int a = S(); // expected-error {{no viable conversion from 'S'}}
      struct S c = b; // expected-error {{no viable conversion from 'struct S'}}
    }
  }
  void g() {
    int S(void);
    struct S { S() {} };
    int a = S();
    struct S b;
    {
      int S(void);
      int a = S();
      struct S c = b;
    }
    {
      struct S { S() {} }; // expected-note {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
      // expected-note@-2 {{candidate constructor (the implicit move constructor) not viable}}
#endif
      int a = S(); // expected-error {{no viable conversion from 'S'}}
      struct S c = b; // expected-error {{no viable conversion from 'struct S'}}
    }
  }

  struct A {
    struct B;
    void f();
    int B;
  };
  struct A::B {};
  void A::f() {
    B = 123;
    struct B b;
  }
}
