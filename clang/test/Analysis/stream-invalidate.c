// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=alpha.unix.Stream \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

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
