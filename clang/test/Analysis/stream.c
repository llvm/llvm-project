// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.Stream -verify %s

#include "Inputs/system-header-simulator.h"

void check_fread() {
  FILE *fp = tmpfile();
  fread(0, 0, 0, fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_fwrite() {
  FILE *fp = tmpfile();
  fwrite(0, 0, 0, fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_fseek() {
  FILE *fp = tmpfile();
  fseek(fp, 0, 0); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_ftell() {
  FILE *fp = tmpfile();
  ftell(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_rewind() {
  FILE *fp = tmpfile();
  rewind(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_fgetpos() {
  FILE *fp = tmpfile();
  fpos_t pos;
  fgetpos(fp, &pos); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_fsetpos() {
  FILE *fp = tmpfile();
  fpos_t pos;
  fsetpos(fp, &pos); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_clearerr() {
  FILE *fp = tmpfile();
  clearerr(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_feof() {
  FILE *fp = tmpfile();
  feof(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_ferror() {
  FILE *fp = tmpfile();
  ferror(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void check_fileno() {
  FILE *fp = tmpfile();
  fileno(fp); // expected-warning {{Stream pointer might be NULL}}
  fclose(fp);
}

void f_open(void) {
  FILE *p = fopen("foo", "r");
  char buf[1024];
  fread(buf, 1, 1, p); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f_seek(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fseek(p, 1, SEEK_SET); // no-warning
  fseek(p, 1, 3); // expected-warning {{The whence argument to fseek() should be SEEK_SET, SEEK_END, or SEEK_CUR}}
  fclose(p);
}

void f_double_close(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fclose(p);
  fclose(p); // expected-warning {{Stream might be already closed}}
}

void f_double_close_alias(void) {
  FILE *p1 = fopen("foo", "r");
  if (!p1)
    return;
  FILE *p2 = p1;
  fclose(p1);
  fclose(p2); // expected-warning {{Stream might be already closed}}
}

void f_use_after_close(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fclose(p);
  clearerr(p); // expected-warning {{Stream might be already closed}}
}

void f_open_after_close(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fclose(p);
  p = fopen("foo", "r");
  if (!p)
    return;
  fclose(p);
}

void f_reopen_after_close(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fclose(p);
  // Allow reopen after close.
  p = freopen("foo", "w", p);
  if (!p)
    return;
  fclose(p);
}

void f_leak(int c) {
  FILE *p = fopen("foo.c", "r");
  if (!p)
    return;
  if(c)
    return; // expected-warning {{Opened stream never closed. Potential resource leak}}
  fclose(p);
}

FILE *f_null_checked(void) {
  FILE *p = fopen("foo.c", "r");
  if (p)
    return p; // no-warning
  else
    return 0;
}

void pr7831(FILE *fp) {
  fclose(fp); // no-warning
}

// PR 8081 - null pointer crash when 'whence' is not an integer constant
void pr8081(FILE *stream, long offset, int whence) {
  fseek(stream, offset, whence);
}

void check_freopen_1() {
  FILE *f1 = freopen("foo.c", "r", (FILE *)0); // expected-warning {{Stream pointer might be NULL}}
  f1 = freopen(0, "w", (FILE *)0x123456);      // Do not report this as error.
}

void check_freopen_2() {
  FILE *f1 = fopen("foo.c", "r");
  if (f1) {
    FILE *f2 = freopen(0, "w", f1);
    if (f2) {
      // Check if f1 and f2 point to the same stream.
      fclose(f1);
      fclose(f2); // expected-warning {{Stream might be already closed.}}
    } else {
      // Reopen failed.
      // f1 is non-NULL but points to a possibly invalid stream.
      rewind(f1); // expected-warning {{Stream might be invalid}}
      // f2 is NULL but the previous error stops the checker.
      rewind(f2);
    }
  }
}

void check_freopen_3() {
  FILE *f1 = fopen("foo.c", "r");
  if (f1) {
    // Unchecked result of freopen.
    // The f1 may be invalid after this call.
    freopen(0, "w", f1);
    rewind(f1); // expected-warning {{Stream might be invalid}}
    fclose(f1);
  }
}

extern FILE *GlobalF;
extern void takeFile(FILE *);

void check_escape1() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fwrite("1", 1, 1, F); // may fail
  GlobalF = F;
  fwrite("1", 1, 1, F); // no warning
}

void check_escape2() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fwrite("1", 1, 1, F); // may fail
  takeFile(F);
  fwrite("1", 1, 1, F); // no warning
}

void check_escape3() {
  FILE *F = tmpfile();
  if (!F)
    return;
  takeFile(F);
  F = freopen(0, "w", F);
  if (!F)
    return;
  fwrite("1", 1, 1, F); // may fail
  fwrite("1", 1, 1, F); // no warning
}

void check_escape4() {
  FILE *F = tmpfile();
  if (!F)
    return;
  fwrite("1", 1, 1, F); // may fail

  // no escape at (non-StreamChecker-handled) system call
  // FIXME: all such calls should be handled by the checker
  fprintf(F, "0");

  fwrite("1", 1, 1, F); // expected-warning {{might be 'indeterminate'}}
  fclose(F);
}

int Test;
_Noreturn void handle_error();

void check_leak_noreturn_1() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  if (Test == 1) {
    handle_error(); // no warning
  }
  rewind(F1);
} // expected-warning {{Opened stream never closed. Potential resource leak}}

// Check that "location uniqueing" works.
// This results in reporting only one occurence of resource leak for a stream.
void check_leak_noreturn_2() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;
  if (Test == 1) {
    return; // expected-warning {{Opened stream never closed. Potential resource leak}}
  }
  rewind(F1);
} // no warning
