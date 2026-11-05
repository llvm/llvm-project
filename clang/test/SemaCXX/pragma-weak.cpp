// RUN: %clang_cc1 -fsyntax-only -verify %s

#pragma weak foo // expected-warning {{weak identifier 'foo' never declared}}
static void foo();
extern "C" {
  void foo() {
  };
}

extern "C" int Test;
#pragma weak test = Test

// The alias declares ::nd; it must not become a redeclaration of N::nd found
// through the using-directive.
namespace N { int nd(int); } // expected-note {{candidate function}}
using namespace N;
extern "C" int __nd(int);
#pragma weak nd = __nd // expected-note {{candidate function}}
extern "C" int __nd(int i) { return i; }
int use_nd() { return nd(1); } // expected-error {{call to 'nd' is ambiguous}}

// The alias redeclares the matching overload.
int ov(int);
int ov(double);
extern "C" int __ov(int);
#pragma weak ov = __ov
extern "C" int __ov(int i) { return i; }
int use_ov() { return ov(1); }
