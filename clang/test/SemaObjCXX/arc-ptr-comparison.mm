// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-arc -verify %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin11 -fsyntax-only -verify -DNOARC %s
#ifdef NOARC
// expected-no-diagnostics
#endif

int testObjCComparisonRules(void *v, id x, id y) {
  return v == x;
#ifndef NOARC
// expected-error@-2 {{implicit conversion of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}}
// expected-note@-3 {{use __bridge to convert directly (no change in ownership)}}
// expected-note@-4 {{use __bridge_retained to make an ARC object available as a +1 'void *'}}
#endif
  return v >= x;
#ifndef NOARC
// expected-error@-2 {{implicit conversion of Objective-C pointer type 'id' to C pointer type 'void *' requires a bridged cast}}
// expected-note@-3 {{use __bridge to convert directly (no change in ownership)}}
// expected-note@-4 {{use __bridge_retained to make an ARC object available as a +1 'void *'}}
#endif
  return v == (id)(void *)0; // OK
  return v == nullptr; // OK
  return v == (void *)0;
  return x == y;
}
