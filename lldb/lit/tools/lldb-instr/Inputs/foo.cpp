#include "foo.h"

#define MACRO_FOO                                                              \
  { void; }
#define MACRO_BAR(B) B

Foo::Foo() {}
void Foo::A() {}
void Foo::B(int i) {}
int Foo::C(int i) { return i; }
int Foo::D(bool b) const { return 1; }
void Foo::E() {}
int Foo::F(int i) { return i; }
void Foo::G(const char *fmt...) {}
Foo Foo::H() { return Foo(); }
void Foo::I() const { MACRO_FOO; }
Bar Foo::J() const { return MACRO_BAR(Bar()); }
