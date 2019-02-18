// RUN: %clang_analyze_cc1 %s -analyzer-checker=core.NullDereference -analyzer-output=text -verify
// RUN: %clang_analyze_cc1 %s -analyzer-checker=core.NullDereference -analyzer-output=plist -o %t
// RUN: cat %t | %diff_plist %S/Inputs/expected-plists/conditional-path-notes.c.plist -

void testCondOp(int *p) {
  int *x = p ? p : p;
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{'?' condition is false}}
  // expected-note@-3 {{'x' initialized to a null pointer value}}
  *x = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'x')}}
}

void testCondProblem(int *p) {
  if (p) return;
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{Taking false branch}}

  int x = *p ? 0 : 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  (void)x;
}

void testLHSProblem(int *p) {
  int x = !p ? *p : 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{'?' condition is true}}
  // expected-note@-3 {{Dereference of null pointer (loaded from variable 'p')}}
  (void)x;
}

void testRHSProblem(int *p) {
  int x = p ? 1 : *p; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{'?' condition is false}}
  // expected-note@-3 {{Dereference of null pointer (loaded from variable 'p')}}
  (void)x;
}

void testBinaryCondOp(int *p) {
  int *x = p ?: p;
  // expected-note@-1 {{'?' condition is false}}
  // expected-note@-2 {{'x' initialized to a null pointer value}}
  *x = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'x')}}
}

void testBinaryLHSProblem(int *p) {
  if (p) return;
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{Taking false branch}}

  int x = *p ?: 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  (void)x;
}

void testDiagnosableBranch(int a) {
  if (a) {
    // expected-note@-1 {{Assuming 'a' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    *(volatile int *)0 = 1; // expected-warning{{Dereference of null pointer}}
    // expected-note@-1 {{Dereference of null pointer}}
  }
}

void testDiagnosableBranchLogical(int a, int b) {
  if (a && b) {
    // expected-note@-1 {{Assuming 'a' is not equal to 0}}
    // expected-note@-2 {{Left side of '&&' is true}}
    // expected-note@-3 {{Assuming 'b' is not equal to 0}}
    // expected-note@-4 {{Taking true branch}}
    *(volatile int *)0 = 1; // expected-warning{{Dereference of null pointer}}
    // expected-note@-1 {{Dereference of null pointer}}
  }
}

void testNonDiagnosableBranchArithmetic(int a, int b) {
  if (a - b) {
    // expected-note@-1 {{Taking true branch}}
    // expected-note@-2 {{Assuming the condition is true}}
    *(volatile int *)0 = 1; // expected-warning{{Dereference of null pointer}}
    // expected-note@-1 {{Dereference of null pointer}}
  }
}

