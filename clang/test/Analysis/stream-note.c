// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.Stream -analyzer-output text -verify %s

#include "Inputs/system-header-simulator.h"

void check_note_at_correct_open() {
  FILE *F1 = tmpfile(); // expected-note {{Stream opened here}}
  if (!F1)
    // expected-note@-1 {{'F1' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
  FILE *F2 = tmpfile();
  if (!F2) {
    // expected-note@-1 {{'F2' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    fclose(F1);
    return;
  }
  rewind(F2);
  fclose(F2);
  rewind(F1);
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_fopen() {
  FILE *F = fopen("file", "r"); // expected-note {{Stream opened here}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_freopen() {
  FILE *F = fopen("file", "r"); // expected-note {{Stream opened here}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
  F = freopen(0, "w", F); // expected-note {{Stream reopened here}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_leak_2(int c) {
  FILE *F1 = fopen("foo1.c", "r"); // expected-note {{Stream opened here}}
  if (!F1)
    // expected-note@-1 {{'F1' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    // expected-note@-3 {{'F1' is non-null}}
    // expected-note@-4 {{Taking false branch}}
    return;
  FILE *F2 = fopen("foo2.c", "r"); // expected-note {{Stream opened here}}
  if (!F2) {
    // expected-note@-1 {{'F2' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    // expected-note@-3 {{'F2' is non-null}}
    // expected-note@-4 {{Taking false branch}}
    fclose(F1);
    return;
  }
  if (c)
    // expected-note@-1 {{Assuming 'c' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'c' is not equal to 0}}
    // expected-note@-4 {{Taking true branch}}
    return;
  // expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
  // expected-note@-2 {{Opened stream never closed. Potential resource leak}}
  // expected-warning@-3 {{Opened stream never closed. Potential resource leak}}
  // expected-note@-4 {{Opened stream never closed. Potential resource leak}}
  fclose(F1);
  fclose(F2);
}
