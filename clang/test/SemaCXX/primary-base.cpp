// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
class A { virtual void f(); };
class B : virtual A { };

class C : B { };

// Since A is already a primary base class, C should be the primary base class
// of F.
class F : virtual A, virtual C { };

int sa[sizeof(F) == sizeof(A) ? 1 : -1];
