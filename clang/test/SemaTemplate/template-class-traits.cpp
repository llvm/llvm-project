// RUN: %clang_cc1 -fsyntax-only -verify %s 
// expected-no-diagnostics
#define T(b) (b) ? 1 : -1
#define F(b) (b) ? -1 : 1

struct HasVirt { virtual void a(); };
template<class T> struct InheritPolymorph : HasVirt {};
int t01[T(__is_polymorphic(InheritPolymorph<int>))];

