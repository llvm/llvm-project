// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify -DUSE_EARLY
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-note@Inputs/def.h:5 {{here}}

@class Def;
Def *def;

@import decldef;
#ifdef USE_EARLY
A *a1; // expected-error{{declaration of 'A' must be imported from module 'decldef.Def' before it is required}}
#endif
B *b1;
#ifdef USE_EARLY
// expected-error@-2{{must use 'struct' tag to refer to type 'B'}}
#else
// expected-error@-4{{declaration of 'B' must be imported from module 'decldef.Decl' before it is required}}
// expected-note@Inputs/decl.h:2 {{not visible}}
#endif
@import decldef.Decl;

A *a2;
struct B *b;

void testA(A *a) {
  a->ivar = 17;
#ifndef USE_EARLY
  // expected-error@-2{{definition of 'A' must be imported from module 'decldef.Def' before it is required}}
#endif
}

void testB(void) {
  B b; // Note: redundant error silenced
}

void testDef(void) {
  [def defMethod];
}
