// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin11 -fsyntax-only -fobjc-runtime-has-weak -fobjc-arc -verify %s
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

@class A;

int testMixedQualComparisonRules(void *v, const void *cv, A *a, const A *ca) {
  return cv == ca;
#ifndef NOARC
// expected-error@-2 {{implicit conversion of Objective-C pointer type 'const A *' to C pointer type 'const void *' requires a bridged cast}}
// expected-note@-3 {{use __bridge to convert directly (no change in ownership)}}
// expected-note@-4 {{use __bridge_retained to make an ARC object available as a +1 'const void *'}}
#endif
  // FIXME: The "to" type in this diagnostic is wrong; we should convert to "const void *".
  return v == ca;
#ifndef NOARC
// expected-error@-2 {{implicit conversion of Objective-C pointer type 'const A *' to C pointer type 'void *' requires a bridged cast}}
// expected-note@-3 {{use __bridge to convert directly (no change in ownership)}}
// expected-note@-4 {{use __bridge_retained to make an ARC object available as a +1 'void *'}}
#endif
  return cv == a;
#ifndef NOARC
// expected-error@-2 {{implicit conversion of Objective-C pointer type 'A *' to C pointer type 'const void *' requires a bridged cast}}
// expected-note@-3 {{use __bridge to convert directly (no change in ownership)}}
// expected-note@-4 {{use __bridge_retained to make an ARC object available as a +1 'const void *'}}
#endif

  // FIXME: Shouldn't these be rejected in ARC mode too?
  return ca == cv;
  return a == cv;
  return ca == v;
}

#ifndef NOARC
int testDoublePtr(void *pv, void **ppv, A *__strong* pspa, A *__weak* pwpa, A *__strong** ppspa) {
  return pv == pspa;
  return pspa == pv;
  return pv == pspa;
  return pv == pwpa;
  return pspa == pwpa; // expected-error {{comparison of distinct pointer types}}
  return ppv == pspa; // expected-error {{comparison of distinct pointer types}}
  return pspa == ppv; // expected-error {{comparison of distinct pointer types}}
  return pv == ppspa;
  return ppv == ppspa; // expected-error{{comparison of distinct pointer types}}
}
#endif
