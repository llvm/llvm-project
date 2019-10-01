// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file  %s -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/deref-track-symbolic-region.c.plist -

struct S {
  int *x;
  int y;
};

int *foo();

void test(struct S syz, int *pp) {
  int m = 0;
  syz.x = foo(); // expected-note{{Value assigned to 'syz.x'}}

  struct S *ps = &syz;
  if (ps->x)
    //expected-note@-1{{Assuming field 'x' is null}}
    //expected-note@-2{{Taking false branch}}

    m++;

  m += *syz.x; // expected-warning{{Dereference of null pointer (loaded from field 'x')}}
  // expected-note@-1{{Dereference of null pointer (loaded from field 'x')}}
}

void testTrackConstraintBRVisitorIsTrackingTurnedOn(struct S syz, int *pp) {
  int m = 0;
  syz.x = foo(); // expected-note{{Value assigned to 'syz.x'}}

  struct S *ps = &syz;
  if (ps->x)
    //expected-note@-1{{Assuming field 'x' is null}}
    //expected-note@-2{{Taking false branch}}

    m++;
  int *p = syz.x; //expected-note {{'p' initialized to a null pointer value}}
  m = *p; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
          // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
}

