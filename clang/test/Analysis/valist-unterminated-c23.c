// RUN: %clang_analyze_cc1 -triple hexagon-unknown-linux -std=c23 -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c23 -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-output=text

// Test that the security.VAList checker detects leaks with __builtin_c23_va_start.

#include "Inputs/system-header-simulator-for-valist-c23.h"

void c23_leak(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  return; // expected-warning{{Initialized va_list 'va' is leaked}}
  // expected-note@-1{{Initialized va_list 'va' is leaked}}
}

void c23_reinit(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
                      // expected-note@-1{{Initialized va_list}}
  va_start(va, fst); // expected-warning{{Initialized va_list 'va' is initialized again}}
  // expected-note@-1{{Initialized va_list 'va' is initialized again}}
} // expected-warning{{Initialized va_list 'va' is leaked}}
  // expected-note@-1{{Initialized va_list 'va' is leaked}}

void c23_reinit_ok(int fst, ...) {
  va_list va;
  va_start(va, fst);
  va_end(va);
  va_start(va, fst);
  va_end(va);
} // no-warning
