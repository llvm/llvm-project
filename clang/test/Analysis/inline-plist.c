// RUN: %clang_analyze_cc1 %s -analyzer-checker=core.NullDereference,core.DivideZero -fblocks -analyzer-output=text -analyzer-config suppress-null-return-paths=false -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -analyzer-config eagerly-assume=false %s -analyzer-checker=core.NullDereference,core.DivideZero -fblocks -analyzer-output=plist -analyzer-config suppress-null-return-paths=false -o %t
// RUN: %normalize_plist <%t | diff -ub %S/Inputs/expected-plists/inline-plist.c.plist -

// <rdar://problem/10967815>
void mmm(int y) {
  if (y != 0)
    y++;
}

int foo(int x, int y) {
  mmm(y);
  if (x != 0) {
    // expected-note@-1 {{Assuming 'x' is equal to 0}}
    // expected-note@-2 {{Taking false branch}}
    x++;
  }
  return 5/x; // expected-warning{{Division by zero}} expected-note{{Division by zero}}
}

// Test a bug triggering only when inlined.
void has_bug(int *p) {
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
}

void test_has_bug(void) {
  has_bug(0);
  // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
  // expected-note@-2 {{Calling 'has_bug'}}
}

void triggers_bug(int *p) {
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
}

// This function triggers a bug by calling triggers_bug().  The diagnostics
// should show when p is assumed to be null.
void bar(int *p) {
  if (!!p) {
    // expected-note@-1 {{Assuming 'p' is null}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (p == 0) {
    // expected-note@-1 {{'p' is equal to null}}
    // expected-note@-2 {{Taking true branch}}
    triggers_bug(p);
    // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
    // expected-note@-2 {{Calling 'triggers_bug'}}
  }
}

// ========================================================================== //
// Test inlining of blocks.
// ========================================================================== //

void test_block__capture_null(void) {
  int *p = 0; // expected-note{{'p' initialized to a null pointer value}}
  ^(void){ // expected-note {{Calling anonymous block}}
    *p = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
  }();

}

void test_block_ret(void) {
  int *p = ^int*(void){ // expected-note {{Calling anonymous block}} expected-note{{Returning to caller}} expected-note {{'p' initialized to a null pointer value}}
    int *q = 0; // expected-note {{'q' initialized to a null pointer value}}
    return q; // expected-note {{Returning null pointer (loaded from 'q')}}
  }();
  *p = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
}

void test_block_blockvar(void) {
  __block int *p;
  ^(void){ // expected-note{{Calling anonymous block}} expected-note{{Returning to caller}}
    p = 0; // expected-note{{Null pointer value stored to 'p'}}
  }();
  *p = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
}

void test_block_arg(void) {
  int *p;
  ^(int **q){ // expected-note{{Calling anonymous block}} expected-note{{Returning to caller}}
    *q = 0; // expected-note{{Null pointer value stored to 'p'}}
  }(&p);
  *p = 1; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}} expected-note{{Dereference of null pointer (loaded from variable 'p')}}
}

