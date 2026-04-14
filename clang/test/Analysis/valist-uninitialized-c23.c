// RUN: %clang_analyze_cc1 -triple hexagon-unknown-linux -std=c23 -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c23 -verify %s \
// RUN:   -analyzer-checker=core,security.VAList \
// RUN:   -analyzer-disable-checker=core.CallAndMessage \
// RUN:   -analyzer-output=text
//
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu -std=c23 %s \
// RUN:   -analyzer-checker=core,security.VAList

// Test that the security.VAList checker recognizes __builtin_c23_va_start,
// which is used when compiling in C23 mode.

#include "Inputs/system-header-simulator-for-valist-c23.h"

void c23_no_warning(int fst, ...) {
  va_list va;
  va_start(va, fst);
  (void)va_arg(va, int);
  va_end(va);
} // no-warning

void c23_one_arg_no_warning(int fst, ...) {
  va_list va;
  va_start(va);
  (void)va_arg(va, int);
  va_end(va);
} // no-warning

void c23_uninitialized(int fst, ...) {
  va_list va;
  (void)va_arg(va, int); // expected-warning{{va_arg() is called on an uninitialized va_list}}
  // expected-note@-1{{va_arg() is called on an uninitialized va_list}}
}

void c23_use_after_end(int fst, ...) {
  va_list va;
  va_start(va, fst); // expected-note{{Initialized va_list}}
  va_end(va); // expected-note{{Ended va_list}}
  (void)va_arg(va, int); // expected-warning{{va_arg() is called on an already released va_list}}
  // expected-note@-1{{va_arg() is called on an already released va_list}}
}

void c23_copy(int fst, ...) {
  va_list va, va2;
  va_start(va, fst);
  va_copy(va2, va);
  va_end(va);
  (void)va_arg(va2, int);
  va_end(va2);
} // no-warning

void c23_vprintf(int isstring, ...) {
  va_list va;
  va_start(va, isstring);
  vprintf(isstring ? "%s" : "%d", va);
  va_end(va);
} // no-warning
