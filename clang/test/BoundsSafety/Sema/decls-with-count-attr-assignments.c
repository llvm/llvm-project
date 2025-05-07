
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  int *__counted_by(l) bp;
  int *bp2 __counted_by(l+1);
  int l;
};

int side_effect(int *ptr);

int Test (int *__counted_by(n) buf, int n, struct S s) {
  // expected-note@+1{{'buf' has been assigned here}}
  buf = 0;
  // expected-error@+1{{cannot reference 'buf' after it is changed during consecutive assignments}}
  n = side_effect(buf); // expected-error{{assignments to dependent variables should not have side effects between them}}

  s.bp = 0;
  // expected-note@+1{{previously assigned here}}
  s.l = side_effect(s.bp2);
  // expected-error@+2{{assignment to 's.l' requires corresponding assignment to 'int *__single __counted_by(l + 1)' (aka 'int *__single') 's.bp2'; add self assignment 's.bp2 = s.bp2' if the value has not changed}}
  // expected-error@+1{{multiple consecutive assignments to a dynamic count 'l' must be simplified; keep only one of the assignments}}
  s.l = 0; // expected-error{{assignments to dependent variables should not have side effects between them}}

  n = side_effect(buf); // no error the side effect (RHS) happens before the assignment
  buf = 0;

  return 0;
}
