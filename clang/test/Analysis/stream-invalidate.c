// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=alpha.unix.Stream \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"
#include "Inputs/system-header-simulator-for-valist.h"

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);

void test_fread(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;

  char Buf[3] = {10, 10, 10};
  fread(Buf, 1, 3, F);
  // The check applies to success and failure.
  clang_analyzer_dump(Buf[0]); // expected-warning {{conj_$}} Should not preserve the previous value, thus should not be 10.
  clang_analyzer_dump(Buf[2]); // expected-warning {{conj_$}}
  if (feof(F)) {
    char Buf1[3] = {10, 10, 10};
    fread(Buf1, 1, 3, F); // expected-warning {{is in EOF state}}
    clang_analyzer_dump(Buf1[0]); // expected-warning {{10 S32b}}
    clang_analyzer_dump(Buf1[2]); // expected-warning {{10 S32b}}
  }

  fclose(F);
}

void test_fwrite(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;

  char Buf[3] = {10, 10, 10};
  fwrite(Buf, 1, 3, F);
  // The check applies to success and failure.
  clang_analyzer_dump(Buf[0]); // expected-warning {{10 S32b}}
  clang_analyzer_dump(Buf[2]); // expected-warning {{10 S32b}}

  fclose(F);
}

void test_fgets() {
  FILE *F = tmpfile();
  if (!F)
    return;

  char Buf[3] = {10, 10, 10};
  fgets(Buf, 3, F);
  // The check applies to success and failure.
  clang_analyzer_dump(Buf[0]); // expected-warning {{conj_$}} Should not preserve the previous value, thus should not be 10.
  clang_analyzer_dump(Buf[2]); // expected-warning {{conj_$}}
  if (feof(F)) {
    char Buf1[3] = {10, 10, 10};
    fgets(Buf1, 3, F); // expected-warning {{is in EOF state}}
    clang_analyzer_dump(Buf1[0]); // expected-warning {{10 S32b}}
    clang_analyzer_dump(Buf1[2]); // expected-warning {{10 S32b}}
  }

  fclose(F);
}

void test_fputs() {
  FILE *F = tmpfile();
  if (!F)
    return;

  char *Buf = "aaa";
  fputs(Buf, F);
  // The check applies to success and failure.
  clang_analyzer_dump(Buf[0]); // expected-warning {{97 S32b}}
  clang_analyzer_dump(Buf[2]); // expected-warning {{97 S32b}}
  clang_analyzer_dump(Buf[3]); // expected-warning {{0 S32b}}

  fclose(F);
}

void test_fscanf() {
  FILE *F = tmpfile();
  if (!F)
    return;

  int a = 1;
  unsigned b;
  int Ret = fscanf(F, "%d %u", &a, &b);
  if (Ret == 0) {
    clang_analyzer_dump(a); // expected-warning {{conj_$}}
    // FIXME: should be {{1 S32b}}.
    clang_analyzer_dump(b); // expected-warning {{conj_$}}
    // FIXME: should be {{uninitialized value}}.
  } else if (Ret == 1) {
    clang_analyzer_dump(a); // expected-warning {{conj_$}}
    clang_analyzer_dump(b); // expected-warning {{conj_$}}
    // FIXME: should be {{uninitialized value}}.
  } else if (Ret >= 2) {
    clang_analyzer_dump(a); // expected-warning {{conj_$}}
    clang_analyzer_dump(b); // expected-warning {{conj_$}}
    clang_analyzer_eval(Ret == 2); // expected-warning {{FALSE}} expected-warning {{TRUE}}
    // FIXME: should be only TRUE.
  } else {
    clang_analyzer_dump(a); // expected-warning {{1 S32b}}
    clang_analyzer_dump(b); // expected-warning {{uninitialized value}}
  }

  fclose(F);
}

void test_getdelim(char *P, size_t Sz) {
  FILE *F = tmpfile();
  if (!F)
    return;

  char *P1 = P;
  size_t Sz1 = Sz;
  ssize_t Ret = getdelim(&P, &Sz, '\t', F);
  if (Ret < 0) {
    clang_analyzer_eval(P == P1); // expected-warning {{FALSE}} \
                                  // expected-warning {{TRUE}}
    clang_analyzer_eval(Sz == Sz1); // expected-warning {{FALSE}} \
                                    // expected-warning {{TRUE}}
  } else {
    clang_analyzer_eval(P == P1); // expected-warning {{FALSE}} \
                                  // expected-warning {{TRUE}}
    clang_analyzer_eval(Sz == Sz1); // expected-warning {{FALSE}} \
                                    // expected-warning {{TRUE}}
  }

  fclose(F);
}

void test_fgetpos() {
  FILE *F = tmpfile();
  if (!F)
    return;

  fpos_t Pos = 1;
  int Ret = fgetpos(F, &Pos);
  if (Ret == 0) {
    clang_analyzer_dump(Pos); // expected-warning {{conj_$}}
  } else {
    clang_analyzer_dump(Pos); // expected-warning {{1 S32b}}
  }

  fclose(F);
}

void test_fprintf() {
  FILE *F1 = tmpfile();
  if (!F1)
    return;

  unsigned a = 42;
  char *output = "HELLO";
  int r = fprintf(F1, "%s\t%u\n", output, a);
  // fprintf does not invalidate any of its input
  // 69 is ascii for 'E'
  clang_analyzer_dump(a); // expected-warning {{42 S32b}}
  clang_analyzer_dump(output[1]); // expected-warning {{69 S32b}}
  fclose(F1);
}

int test_vfscanf_inner(const char *fmt, ...) {
  FILE *F1 = tmpfile();
  if (!F1)
    return EOF;

  va_list ap;
  va_start(ap, fmt);

  int r = vfscanf(F1, fmt, ap);

  fclose(F1);
  va_end(ap);
  return r;
}

void test_vfscanf() {
  int i = 42;
  int j = 43;
  int r = test_vfscanf_inner("%d", &i);
  if (r != EOF) {
    // i gets invalidated by the call to test_vfscanf_inner, not by vfscanf.
    clang_analyzer_dump(i); // expected-warning {{conj_$}}
    clang_analyzer_dump(j); // expected-warning {{43 S32b}}
  }
}
