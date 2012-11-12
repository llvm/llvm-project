// RUN: %clang_cc1 -fsyntax-only -Wno-error=address-of-temporary -verify -std=gnu++11 %s
struct X {
  X();
  X(int);
  X(int, int);
};

void f0() { (void)&X(); } // expected-warning{{taking the address of a temporary object}}
void f1() { (void)&X(1); } // expected-warning{{taking the address of a temporary object}}
void f2() { (void)&X(1, 2); } // expected-warning{{taking the address of a temporary object}}
void f3() { (void)&(X)1; } // expected-warning{{taking the address of a temporary object}}


namespace PointerToArrayDecay {
  struct Y {
    int a[4];
  };

  typedef int A[4];

  template<typename T> void consume(T);
  struct S { int *p; };

  void g0() { int *p = Y().a; } // expected-warning{{pointer is initialized by a temporary array}}
  void g1() { int *p = Y{}.a; } // expected-warning{{pointer is initialized by a temporary array}}
  void g2() { int *p = A{}; } // expected-warning{{pointer is initialized by a temporary array}}
  void g3() { int *p = (A){}; } // expected-warning{{pointer is initialized by a temporary array}}

  void h0() { consume(Y().a); }
  void h1() { consume(Y{}.a); }
  void h2() { consume(A{}); }
  void h3() { consume((A){}); }

  void i0() { S s = { Y().a }; } // expected-warning{{pointer is initialized by a temporary array}}
  void i1() { S s = { Y{}.a }; } // expected-warning{{pointer is initialized by a temporary array}}
  void i2() { S s = { A{} }; } // expected-warning{{pointer is initialized by a temporary array}}
  void i3() { S s = { (A){} }; } // expected-warning{{pointer is initialized by a temporary array}}

  void j0() { (void)S { Y().a }; }
  void j1() { (void)S { Y{}.a }; }
  void j2() { (void)S { A{} }; }
  void j3() { (void)S { (A){} }; }

  void k0() { consume(S { Y().a }); }
  void k1() { consume(S { Y{}.a }); }
  void k2() { consume(S { A{} }); }
  void k3() { consume(S { (A){} }); }
}
