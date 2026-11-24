// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -analyzer-output=text -verify %s

// Test that diagnostics cross the implicit cleanup call: the path describes
// the call at the end of the variable's scope, and findings inside and around
// the inlined cleanup frame are reported.

#include "Inputs/system-header-simulator-for-malloc.h"

//===----------------------------------------------------------------------===//
// A null dereference inside the cleanup body: the path enters the inlined
// cleanup through a note anchored at the closing brace of its scope.
//===----------------------------------------------------------------------===//

static void deref_cleanup(int **p) {
  **p = 1; // expected-warning {{Dereference of null pointer}}
           // expected-note@-1 {{Dereference of null pointer}}
}

void null_deref_in_cleanup(void) {
  int *p __attribute__((cleanup(deref_cleanup)));
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
} // expected-note {{Calling 'deref_cleanup'}}

//===----------------------------------------------------------------------===//
// A leak through a no-op inlined cleanup frame: the report survives the
// inlined cleanup call.
//===----------------------------------------------------------------------===//

static void empty_cleanup(int **p) { (void)p; }

void leak_through_cleanup_frame(void) {
  int *p __attribute__((cleanup(empty_cleanup)));
  p = malloc(10); // expected-note {{Memory is allocated}}
} // expected-warning {{Potential leak of memory pointed to by 'p'}}
  // expected-note@-1 {{Potential leak of memory pointed to by 'p'}}

//===----------------------------------------------------------------------===//
// With the attribute split across lines, the "Calling ..." note is still
// anchored at the closing brace of the scope, not at the attribute or the
// function name on a different line.
//===----------------------------------------------------------------------===//

static void multiline_cleanup(int **p) {
  **p = 1; // expected-warning {{Dereference of null pointer}}
           // expected-note@-1 {{Dereference of null pointer}}
}

void null_deref_in_multiline_cleanup_attr(void) {
  int *p __attribute__((cleanup(
      multiline_cleanup)));
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
} // expected-note {{Calling 'multiline_cleanup'}}
