// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -analyzer-output=text -verify %s

// Diagnostics cross the implicit cleanup call; the calling note is anchored at
// the end of the variable's scope.

#include "Inputs/system-header-simulator-for-malloc.h"

// A null dereference in the cleanup body.
static void deref_cleanup(int **p) {
  **p = 1; // expected-warning {{Dereference of null pointer}}
           // expected-note@-1 {{Dereference of null pointer}}
}

void null_deref_in_cleanup(void) {
  int *p __attribute__((cleanup(deref_cleanup)));
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
} // expected-note {{Calling 'deref_cleanup'}}

// A leak survives the inlined cleanup frame.
static void empty_cleanup(int **p) { (void)p; }

void leak_through_cleanup_frame(void) {
  int *p __attribute__((cleanup(empty_cleanup)));
  p = malloc(10); // expected-note {{Memory is allocated}}
} // expected-warning {{Potential leak of memory pointed to by 'p'}}
  // expected-note@-1 {{Potential leak of memory pointed to by 'p'}}

// The attribute spans lines; the calling note stays at the closing brace.
static void multiline_cleanup(int **p) {
  **p = 1; // expected-warning {{Dereference of null pointer}}
           // expected-note@-1 {{Dereference of null pointer}}
}

void null_deref_in_multiline_cleanup_attr(void) {
  int *p __attribute__((cleanup(
      multiline_cleanup)));
  p = 0; // expected-note {{Null pointer value stored to 'p'}}
} // expected-note {{Calling 'multiline_cleanup'}}
