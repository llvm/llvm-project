// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -analyzer-config graph-trim-interval=5 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=plist-multi-file -analyzer-config graph-trim-interval=5 %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/eager-reclamation-path-notes.c.plist

void use(int *ptr, int val) {
  *ptr = val; // expected-warning {{Dereference of null pointer (loaded from variable 'ptr')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'ptr')}}
}

int compute() {
  // Do something that will take enough processing to trigger trimming.
  // FIXME: This is actually really sensitive. If the interval timing is just
  // wrong, the node for the actual dereference may also be collected, and all
  // the path notes will disappear. <rdar://problem/12511814>
  return 2 + 3 + 4 + 5 + 6;
}

void testSimple() {
  int *p = 0;
  // expected-note@-1 {{'p' initialized to a null pointer value}}
  use(p, compute());
  // expected-note@-1 {{Passing null pointer value via 1st parameter 'ptr'}}
  // expected-note@-2 {{Calling 'use'}}
}


void use2(int *ptr, int val) {
  *ptr = val; // expected-warning {{Dereference of null pointer (loaded from variable 'ptr')}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'ptr')}}
}

void passThrough(int *p) {
  use2(p, compute());
  // expected-note@-1 {{Passing null pointer value via 1st parameter 'ptr'}}
  // expected-note@-2 {{Calling 'use2'}}
}

void testChainedCalls() {
  int *ptr = 0;
  // expected-note@-1 {{'ptr' initialized to a null pointer value}}
  passThrough(ptr);
  // expected-note@-1 {{Passing null pointer value via 1st parameter 'p'}}
  // expected-note@-2 {{Calling 'passThrough'}}
}

