// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

class Base { // expected-warning{{class 'Base' does not declare any constructor to initialize its non-modifiable members}}
#if __cplusplus <= 199711L
// expected-error@-2 {{cannot define the implicit copy assignment operator for 'Base', because non-static reference member 'ref' cannot use copy assignment operator}}
#endif

  int &ref; // expected-note{{reference member 'ref' will never be initialized}}
#if __cplusplus <= 199711L
  // expected-note@-2 {{declared here}}
#else
  // expected-note@-4 2 {{copy assignment operator of 'Base' is implicitly deleted because field 'ref' is of reference type 'int &'}}
#endif
};

class X  : Base {
#if __cplusplus <= 199711L
// expected-note@-2 {{assignment operator for 'Base' first required here}}
// expected-error@-3 {{cannot define the implicit copy assignment operator for 'X', because non-static const member 'cint' cannot use copy assignment operator}}
#else
// expected-note@-5 2 {{copy assignment operator of 'X' is implicitly deleted because base class 'Base' has a deleted copy assignment operator}}
#endif

public: 
  X();
  const int cint;
#if __cplusplus <= 199711L
// expected-note@-2 {{declared here}}
#endif
}; 

struct Y  : X { 
  Y();
  Y& operator=(const Y&);
  Y& operator=(volatile Y&);
  Y& operator=(const volatile Y&);
  Y& operator=(Y&);
}; 

class Z : Y {};

Z z1;
Z z2;

// Test1
void f(X x, const X cx) {
  x = cx;
#if __cplusplus <= 199711L
  // expected-note@-2 2{{assignment operator for 'X' first required here}}
#else
  // expected-error@-4 {{object of type 'X' cannot be assigned because its copy assignment operator is implicitly deleted}}
#endif

  x = cx;
#if __cplusplus >= 201103L
  // expected-error@-2 {{object of type 'X' cannot be assigned because its copy assignment operator is implicitly deleted}}
#endif
  z1 = z2;
}

// Test2
class T {};
T t1;
T t2;

void g() {
  t1 = t2;
}

// Test3
class V {
public:
  V();
  V &operator = (V &b);
};

class W : V {};
W w1, w2;

void h() {
  w1 = w2;
}

// Test4

class B1 {
public:
  B1();
  B1 &operator = (B1 b);
};

class D1 : B1 {};
D1 d1, d2;

void i() {
  d1 = d2;
}

// Test5

class E1 {
#if __cplusplus <= 199711L
// expected-error@-2 {{cannot define the implicit copy assignment operator for 'E1', because non-static const member 'a' cannot use copy assignment operator}}
#endif

public:
  const int a;
#if __cplusplus <= 199711L
// expected-note@-2 {{declared here}}
#else
// expected-note@-4 {{copy assignment operator of 'E1' is implicitly deleted because field 'a' is of const-qualified type 'const int'}}
#endif
  E1() : a(0) {}

};

E1 e1, e2;

void j() {
  e1 = e2;
#if __cplusplus <= 199711L
  // expected-note@-2 {{assignment operator for 'E1' first required here}}
#else
  // expected-error@-4 {{object of type 'E1' cannot be assigned because its copy assignment operator is implicitly deleted}}
#endif
}

namespace ProtectedCheck {
  struct X {
  protected:
    X &operator=(const X&);
#if __cplusplus <= 199711L
    // expected-note@-2 {{declared protected here}}
#endif
  };

  struct Y : public X { };

  void f(Y y) { y = y; }

  struct Z {
#if __cplusplus <= 199711L
  // expected-error@-2 {{'operator=' is a protected member of 'ProtectedCheck::X'}}
#endif
    X x;
#if __cplusplus >= 201103L
    // expected-note@-2 {{copy assignment operator of 'Z' is implicitly deleted because field 'x' has an inaccessible copy assignment operator}}
#endif
  };

  void f(Z z) { z = z; }
#if __cplusplus <= 199711L
  // expected-note@-2 {{implicit copy assignment operator}}
#else
  // expected-error@-4 {{object of type 'Z' cannot be assigned because its copy assignment operator is implicitly deleted}}
#endif
}

namespace MultiplePaths {
  struct X0 { 
    X0 &operator=(const X0&);
  };

  struct X1 : public virtual X0 { };

  struct X2 : X0, X1 { }; // expected-warning{{direct base 'X0' is inaccessible due to ambiguity:\n    struct MultiplePaths::X2 -> X0\n    struct MultiplePaths::X2 -> X1 -> X0}}

  void f(X2 x2) { x2 = x2; }
}
