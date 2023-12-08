// RUN: %clang_cc1 -emit-llvm-only %s -verify
// expected-no-diagnostics
// PR5730

struct A { operator int(); float y; };
struct B : A { double z; };
void a() { switch(B()) {} }

