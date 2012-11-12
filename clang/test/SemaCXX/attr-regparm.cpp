// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple i686-apple-darwin10 %s

// PR7025
struct X0 {
  void __attribute__((regparm(3))) f0();
  void __attribute__((regparm(3))) f1();
  void __attribute__((regparm(3))) f2(); // expected-note{{previous declaration is here}}
  void f3(); // expected-note{{previous declaration is here}}
};

void X0::f0() { }
void __attribute__((regparm(3))) X0::f1() { }
void __attribute__((regparm(2))) X0::f2() { } // expected-error{{function declared with with regparm(2) attribute was previously declared with the regparm(3) attribute}}
void __attribute__((regparm(2))) X0::f3() { } // expected-error{{function declared with with regparm(2) attribute was previously declared without the regparm attribute}}
