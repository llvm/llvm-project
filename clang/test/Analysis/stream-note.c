// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Stream -analyzer-output text \
// RUN:   -analyzer-config unix.Stream:Pedantic=true \
// RUN:   -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Stream,unix.StdCLibraryFunctions -analyzer-output text \
// RUN:   -analyzer-config unix.Stream:Pedantic=true \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true -verify=expected,stdargs %s

#include "Inputs/system-header-simulator.h"

void check_note_at_correct_open(void) {
  FILE *F1 = tmpfile(); // expected-note {{Stream opened here}}
  // stdargs-note@-1 {{'tmpfile' is successful}}
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
  // stdargs-note@-1 {{'fopen' is successful}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_freopen(void) {
  FILE *F = fopen("file", "r"); // expected-note {{Stream opened here}}
  // stdargs-note@-1 {{'fopen' is successful}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
  F = freopen(0, "w", F); // expected-note {{Stream reopened here}}
  // stdargs-note@-1 {{'freopen' is successful}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_fdopen(int fd) {
  FILE *F = fdopen(fd, "r"); // expected-note {{Stream opened here}}
  // stdargs-note@-1 {{'fdopen' is successful}}
  if (!F)
    // expected-note@-1 {{'F' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    return;
}
// expected-warning@-1 {{Opened stream never closed. Potential resource leak}}
// expected-note@-2 {{Opened stream never closed. Potential resource leak}}

void check_note_leak_2(int c) {
  FILE *F1 = fopen("foo1.c", "r"); // expected-note {{Stream opened here}}
  // stdargs-note@-1 {{'fopen' is successful}}
  if (!F1)
    // expected-note@-1 {{'F1' is non-null}}
    // expected-note@-2 {{Taking false branch}}
    // expected-note@-3 {{'F1' is non-null}}
    // expected-note@-4 {{Taking false branch}}
    return;
  FILE *F2 = fopen("foo2.c", "r"); // expected-note {{Stream opened here}}
  // stdargs-note@-1 {{'fopen' is successful}}
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
  // stdargs-note@-1 {{'fopen' fails}}
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

void check_indeterminate_notes(void) {
  FILE *F;
  F = fopen("foo1.c", "r");
  if (F == NULL)     // expected-note {{Taking false branch}} \
                     // expected-note {{'F' is not equal to NULL}}
    return;
  int R = fgetc(F);  // no note
  if (R >= 0) {      // expected-note {{Taking true branch}} \
                     // expected-note {{'R' is >= 0}}
    fgetc(F);        // expected-note {{Assuming this stream operation fails}}
    if (ferror(F))   // expected-note {{Taking true branch}}
      fgetc(F);      // expected-warning {{File position of the stream might be 'indeterminate' after a failed operation. Can cause undefined behavior}} \
                     // expected-note {{File position of the stream might be 'indeterminate' after a failed operation. Can cause undefined behavior}}
  }
  fclose(F);
}

void check_indeterminate_after_clearerr(void) {
  FILE *F;
  char Buf[10];
  F = fopen("foo1.c", "r");
  if (F == NULL)          // expected-note {{Taking false branch}} \
                          // expected-note {{'F' is not equal to NULL}}
    return;
  fread(Buf, 1, 1, F);    // expected-note {{Assuming this stream operation fails}}
  if (ferror(F)) {        // expected-note {{Taking true branch}}
    clearerr(F);
    fread(Buf, 1, 1, F);  // expected-warning {{might be 'indeterminate' after a failed operation}} \
                          // expected-note {{might be 'indeterminate' after a failed operation}}
  }
  fclose(F);
}

void check_indeterminate_eof(void) {
  FILE *F;
  char Buf[2];
  F = fopen("foo1.c", "r");
  if (F == NULL)               // expected-note {{Taking false branch}} \
                               // expected-note {{'F' is not equal to NULL}} \
                               // expected-note {{Taking false branch}} \
                               // expected-note {{'F' is not equal to NULL}}
    return;
  fgets(Buf, sizeof(Buf), F);  // expected-note {{Assuming this stream operation fails}} \
                               // expected-note {{Assuming stream reaches end-of-file here}}

  fgets(Buf, sizeof(Buf), F);  // expected-warning {{might be 'indeterminate'}} \
                               // expected-note {{might be 'indeterminate'}} \
                               // expected-warning {{stream is in EOF state}} \
                               // expected-note {{stream is in EOF state}}
  fclose(F);
}

void check_indeterminate_fseek(void) {
  FILE *F = fopen("file", "r");
  if (!F)                           // expected-note {{Taking false branch}} \
                                    // expected-note {{'F' is non-null}}
    return;
  int Ret = fseek(F, 1, SEEK_SET);  // expected-note {{Assuming this stream operation fails}}
  if (Ret) {                        // expected-note {{Taking true branch}} \
                                    // expected-note {{'Ret' is -1}}
    char Buf[2];
    fwrite(Buf, 1, 2, F);           // expected-warning {{might be 'indeterminate'}} \
                                    // expected-note {{might be 'indeterminate'}}
  }
  fclose(F);
}

void error_fseek_ftell(void) {
  FILE *F = fopen("file", "r");
  if (!F)                 // expected-note {{Taking false branch}} \
                          // expected-note {{'F' is non-null}}
    return;
  fseek(F, 0, SEEK_END);  // expected-note {{Assuming this stream operation fails}}
  long size = ftell(F);   // expected-warning {{might be 'indeterminate'}} \
                          // expected-note {{might be 'indeterminate'}}
  if (size == -1) {
    fclose(F);
    return;
  }
  if (size == 1)
    fprintf(F, "abcd");
  fclose(F);
}

void error_fseek_read_eof(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  if (fseek(F, 22, SEEK_SET) == -1) {
    fclose(F);
    return;
  }
  fgetc(F); // no warning
  fclose(F);
}

void check_note_at_use_after_close(void) {
  FILE *F = tmpfile();
  if (!F) // expected-note {{'F' is non-null}} expected-note {{Taking false branch}}
    return;
  fclose(F); // expected-note {{Stream is closed here}}
  rewind(F); // expected-warning {{Use of a stream that might be already closed}}
  // expected-note@-1 {{Use of a stream that might be already closed}}
}
