// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x objective-c -verify -fmodules-cache-path=%t -I %S/Inputs %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -x objective-c++ -verify -fmodules-cache-path=%t -I %S/Inputs %s -DLOCAL_VISIBILITY

// This test checks some of the same things as macros.c, but imports modules in
// a different order.

@import macros_other;

int n0 = TOP_OTHER_DEF_RIGHT_UNDEF; // ok

@import macros_top;

TOP_OTHER_DEF_RIGHT_UNDEF *n0b; // expected-warning{{ambiguous expansion of macro 'TOP_OTHER_DEF_RIGHT_UNDEF'}}
// expected-note@macros_top.h:22 {{expanding this definition of 'TOP_OTHER_DEF_RIGHT_UNDEF'}}
// expected-note@macros_other.h:6 {{other definition of 'TOP_OTHER_DEF_RIGHT_UNDEF'}}

@import macros_right;
@import macros_left;

#ifdef TOP_LEFT_UNDEF
#  error TOP_LEFT_UNDEF should not be defined
#endif

#ifndef TOP_RIGHT_UNDEF
#  error TOP_RIGHT_UNDEF should still be defined
#endif

void test(void) {
  float f;
  TOP_RIGHT_REDEF *fp = &f; // ok, right's definition overrides top's definition

  // Note, left's definition wins here, whereas right's definition wins in
  // macros.c.
  int i;
  LEFT_RIGHT_IDENTICAL *ip = &i;
  LEFT_RIGHT_DIFFERENT *ip2 = &f; // expected-warning{{ambiguous expansion of macro 'LEFT_RIGHT_DIFFERENT'}}
  // expected-note@macros_left.h:14 {{expanding this}}
  // expected-note@macros_right.h:12 {{other}}
  LEFT_RIGHT_DIFFERENT2 *ip3 = &f; // expected-warning{{ambiguous expansion of macro 'LEFT_RIGHT_DIFFERENT2}}
  // expected-note@macros_left.h:11 {{expanding this}}
  // expected-note@macros_right.h:13 {{other}}
#undef LEFT_RIGHT_DIFFERENT3
  int LEFT_RIGHT_DIFFERENT3;
}

@import macros_right.undef;

// See macros.c.
#ifdef LOCAL_VISIBILITY
# ifndef TOP_RIGHT_UNDEF
#  error TOP_RIGHT_UNDEF should still be defined
# endif
#else
# ifdef TOP_RIGHT_UNDEF
#  error TOP_RIGHT_UNDEF should not be defined
# endif
#endif

#ifndef TOP_OTHER_UNDEF1
# error TOP_OTHER_UNDEF1 should still be defined
#endif

#ifndef TOP_OTHER_UNDEF2
# error TOP_OTHER_UNDEF2 should still be defined
#endif

#ifndef TOP_OTHER_REDEF1
# error TOP_OTHER_REDEF1 should still be defined
#endif
int n1 = TOP_OTHER_REDEF1; // expected-warning{{ambiguous expansion of macro 'TOP_OTHER_REDEF1'}}
// expected-note@macros_top.h:19 {{expanding this definition}}
// expected-note@macros_other.h:4 {{other definition}}

#ifndef TOP_OTHER_REDEF2
# error TOP_OTHER_REDEF2 should still be defined
#endif
int n2 = TOP_OTHER_REDEF2; // ok

int n3 = TOP_OTHER_DEF_RIGHT_UNDEF; // ok

int top_redef_in_submodules = TOP_REDEF_IN_SUBMODULES;
@import macros_top.c;
void test2(void) {
  int TOP_REDEF_IN_SUBMODULES = top_redef_in_submodules;
}
