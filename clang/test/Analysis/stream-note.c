// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.Stream -analyzer-output text -verify %s

#include "Inputs/system-header-simulator.h"

void check_note_at_correct_open(void) {
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

void check_note_fopen(void) {
  FILE *F = fopen("file", "r"); // expected-note {{Stream opened here}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_freopen(void) {
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

void check_track_null(void) {
  FILE *F;
  F = fopen("foo1.c", "r"); // expected-note {{Value assigned to 'F'}} expected-note {{Assuming pointer value is null}}
  if (F != NULL) {          // expected-note {{Taking false branch}} expected-note {{'F' is equal to NULL}}
    fclose(F);
    return;
  }
  fclose(F); // expected-warning {{Stream pointer might be NULL}}
  // expected-note@-1 {{Stream pointer might be NULL}}
}

void check_eof_notes_feof_after_feof(void) {
  FILE *F;
  char Buf[10];
  F = fopen("foo1.c", "r");
  if (F == NULL) { // expected-note {{Taking false branch}} expected-note {{'F' is not equal to NULL}}
    return;
  }
  fread(Buf, 1, 1, F);
  if (feof(F)) { // expected-note {{Taking true branch}}
    clearerr(F);
    fread(Buf, 1, 1, F);   // expected-note {{Assuming stream reaches end-of-file here}}
    if (feof(F)) {         // expected-note {{Taking true branch}}
      fread(Buf, 1, 1, F); // expected-warning {{Read function called when stream is in EOF state. Function has no effect}}
      // expected-note@-1 {{Read function called when stream is in EOF state. Function has no effect}}
    }
  }
  fclose(F);
}

void check_eof_notes_feof_after_no_feof(void) {
  FILE *F;
  char Buf[10];
  F = fopen("foo1.c", "r");
  if (F == NULL) { // expected-note {{Taking false branch}} expected-note {{'F' is not equal to NULL}}
    return;
  }
  fread(Buf, 1, 1, F);
  if (feof(F)) { // expected-note {{Taking false branch}}
    fclose(F);
    return;
  } else if (ferror(F)) { // expected-note {{Taking false branch}}
    fclose(F);
    return;
  }
  fread(Buf, 1, 1, F);   // expected-note {{Assuming stream reaches end-of-file here}}
  if (feof(F)) {         // expected-note {{Taking true branch}}
    fread(Buf, 1, 1, F); // expected-warning {{Read function called when stream is in EOF state. Function has no effect}}
    // expected-note@-1 {{Read function called when stream is in EOF state. Function has no effect}}
  }
  fclose(F);
}

void check_eof_notes_feof_or_no_error(void) {
  FILE *F;
  char Buf[10];
  F = fopen("foo1.c", "r");
  if (F == NULL) // expected-note {{Taking false branch}} expected-note {{'F' is not equal to NULL}}
    return;
  int RRet = fread(Buf, 1, 1, F); // expected-note {{Assuming stream reaches end-of-file here}}
  if (ferror(F)) {                // expected-note {{Taking false branch}}
  } else {
    fread(Buf, 1, 1, F); // expected-warning {{Read function called when stream is in EOF state. Function has no effect}}
    // expected-note@-1 {{Read function called when stream is in EOF state. Function has no effect}}
  }
  fclose(F);
}
