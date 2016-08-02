// RUN: %clang_cc1 -fobjc-arc -isystem %S/Inputs %s -DNO_USE
// RUN: %clang_cc1 -fobjc-arc -isystem %S/Inputs %s -verify

#include <arc-system-header.h>

#ifndef NO_USE
void test(id op, void *cp) {
  cp = test0(op); // expected-error {{'test0' is unavailable in ARC}}
  cp = *test1(&op); // expected-error {{'test1' is unavailable in ARC}}
// expected-note@arc-system-header.h:1 {{inline function performs a conversion which is forbidden in ARC}}
// expected-note@arc-system-header.h:5 {{inline function performs a conversion which is forbidden in ARC}}
}

void test3(struct Test3 *p) {
  p->field = 0; // expected-error {{'field' is unavailable in ARC}}
                // expected-note@arc-system-header.h:14 {{declaration uses type that is ill-formed in ARC}}
}

void test4(Test4 *p) {
  p->field1 = 0; // expected-error {{'field1' is unavailable in ARC}}
                 // expected-note@arc-system-header.h:19 {{declaration uses type that is ill-formed in ARC}}
  p->field2 = 0;
}

void test5(struct Test5 *p) {
  p->field = 0; // expected-error {{'field' is unavailable in ARC}}
                // expected-note@arc-system-header.h:25 {{field has non-trivial ownership qualification}}
}

id test6() {
  // This is actually okay to use if declared in a system header.
  id x;
  x = (id) kMagicConstant;
  x = (id) (x ? kMagicConstant : kMagicConstant);
  x = (id) (x ? kMagicConstant : (void*) 0);

  extern void test6_helper();
  x = (id) (test6_helper(), kMagicConstant);
}

void test7(Test7 *p) {
  *p.prop = 0; // expected-error {{'prop' is unavailable in ARC}}
  p.prop = 0; // expected-error {{'prop' is unavailable in ARC}}
  *[p prop] = 0; // expected-error {{'prop' is unavailable in ARC}}
  [p setProp: 0]; // expected-error {{'setProp:' is unavailable in ARC}}
// expected-note@arc-system-header.h:41 4 {{declaration uses type that is ill-formed in ARC}}
// expected-note@arc-system-header.h:41 2 {{property 'prop' is declared unavailable here}}
}
#endif

// test8 in header
