// RUN: %clang_cc1 -std=c++11 -verify %s
struct A { // expected-note 2{{candidate}}
  A(int); // expected-note {{candidate}}
  int n;
};
int a = A().n; // expected-error {{no matching constructor}}

struct B {
  B() = delete; // expected-note 4{{here}}
  int n;
};
int b = B().n; // expected-error {{call to deleted}}

struct C {
  B b; // expected-note {{deleted default constructor}}
};
int c = C().b.n; // expected-error {{call to implicitly-deleted default}}

struct D {
  D() = default; // expected-note {{here}} expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  B b; // expected-note 2{{'b' has a deleted default constructor}}
};
int d = D().b.n; // expected-error {{call to implicitly-deleted default}}

struct E {
  E() = default;
  int n;
};
int e = E().n; // ok

struct F {
  F();
  int n;
};
int f = F().n; // ok

union G {
  F f; // expected-note {{non-trivial default constructor}}
};
int g = G().f.n; // expected-error {{call to implicitly-deleted default}}

struct H {
  int n;
private:
  H(); // expected-note {{here}}
};
int h = H().n; // expected-error {{private constructor}}

struct I {
  H h; // expected-note {{inaccessible default constructor}}
};
int i = I().h.n; // expected-error {{call to implicitly-deleted default}}

struct J {
  J();
  virtual int f();
  int n;
};
int j1 = J().n; // ok
int j2 = J().f(); // ok

union K {
  J j; // expected-note 2{{non-trivial default constructor}}
  int m;
};
int k1 = K().j.n; // expected-error {{call to implicitly-deleted default}}
int k2 = K().j.f(); // expected-error {{call to implicitly-deleted default}}
