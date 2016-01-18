// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify -DUSE_EARLY
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-note@Inputs/def.h:5 {{previous}}

@class Def;
Def *def;

@import decldef;
#ifdef USE_EARLY
A *a1; // expected-error{{declaration of 'A' must be imported from module 'decldef.Def' before it is required}}
#endif
B *b1; // expected-error{{must use 'struct' tag to refer to type 'B'}}
@import decldef.Decl;

A *a2;
struct B *b;

void testA(A *a) {
  a->ivar = 17;
#ifndef USE_EARLY
  // expected-error@-2{{definition of 'A' must be imported from module 'decldef.Def' before it is required}}
#endif
}

void testB() {
  B b; // Note: redundant error silenced
}

void testDef() {
  [def defMethod];
}
