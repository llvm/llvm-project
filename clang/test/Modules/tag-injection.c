// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'struct a;' > %t/a.h
// RUN: echo 'struct b {}; void foo(struct b*);' > %t/b.h
// RUN: echo 'module X { module a { header "a.h" } module b { header "b.h" } }' > %t/x.modulemap
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/x.modulemap %s -I%t -verify

#include "a.h"

void f(struct a *p);

// FIXME: We should warn that 'b' will not be visible outside of this function,
// but we merge this 'b' with X.b's 'b' because we don't yet implement C's
// "compatible types" rule.
void g(struct b *p);

struct b b; // expected-error {{definition of 'b' must be imported from module 'X.b' before it is required}}
// expected-note@b.h:1 {{here}}
