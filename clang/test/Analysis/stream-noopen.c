// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -analyzer-checker=alpha.unix.Stream \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=debug.ExprInspection

// enable only StdCLibraryFunctions checker
// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Errno \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"
#include "Inputs/errno_var.h"

void clang_analyzer_eval(int);

const char *WBuf = "123456789";
char RBuf[10];

void test_freopen(FILE *F) {
  F = freopen("xxx", "w", F);
  if (F) {
    if (errno) {} // expected-warning{{undefined}}
  } else {
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
}

void test_fread(FILE *F) {
  size_t Ret = fread(RBuf, 1, 10, F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == 10) {
    if (errno) {} // expected-warning{{undefined}}
  } else {
    clang_analyzer_eval(Ret < 10); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_fwrite(FILE *F) {
  size_t Ret = fwrite(WBuf, 1, 10, F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == 10) {
    if (errno) {} // expected-warning{{undefined}}
  } else {
    clang_analyzer_eval(Ret < 10); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_fgetc(FILE *F) {
  int Ret = fgetc(F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret != EOF) {
    if (errno) {} // expected-warning {{undefined}}
  } else {
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
                                     // expected-warning@-1 {{FALSE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_fputc(FILE *F) {
  int Ret = fputc('a', F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret != EOF) {
    clang_analyzer_eval(Ret == 'a'); // expected-warning {{TRUE}}
    if (errno) {} // expected-warning {{undefined}}
  } else {
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_fgets(char *Buf, int N, FILE *F) {
  char *Ret = fgets(Buf, N, F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  clang_analyzer_eval(Buf != NULL); // expected-warning {{TRUE}}
  clang_analyzer_eval(N >= 0); // expected-warning {{TRUE}}
  if (Ret == Buf) {
    if (errno) {} // expected-warning {{undefined}}
  } else {
    clang_analyzer_eval(Ret == 0); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
                                     // expected-warning@-1 {{FALSE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}

  char Buf1[10];
  Ret = fgets(Buf1, 11, F); // expected-warning {{The 1st argument to 'fgets' is a buffer with size 10}}
}

void test_fgets_bufsize(FILE *F) {
  char Buf[10];
  fgets(Buf, 11, F); // expected-warning {{The 1st argument to 'fgets' is a buffer with size 10}}
}

void test_fputs(char *Buf, FILE *F) {
  int Ret = fputs(Buf, F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  clang_analyzer_eval(Buf != NULL); // expected-warning {{TRUE}}
  if (Ret >= 0) {
    if (errno) {} // expected-warning {{undefined}}
  } else {
    clang_analyzer_eval(Ret == EOF); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_ungetc(FILE *F) {
  int Ret = ungetc('X', F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == 'X') {
    if (errno) {} // expected-warning {{undefined}}
  } else {
    clang_analyzer_eval(Ret == EOF); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_ungetc_EOF(FILE *F, int C) {
  int Ret = ungetc(EOF, F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  clang_analyzer_eval(Ret == EOF); // expected-warning {{TRUE}}
  clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  Ret = ungetc(C, F);
  if (Ret == EOF) {
    clang_analyzer_eval(C == EOF); // expected-warning {{TRUE}}
                                   // expected-warning@-1{{FALSE}}
  }
}

void test_fclose(FILE *F) {
  int Ret = fclose(F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == 0) {
    if (errno) {} // expected-warning{{undefined}}
  } else {
    clang_analyzer_eval(Ret == EOF); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_fseek(FILE *F) {
  int Ret = fseek(F, SEEK_SET, 1);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == 0) {
    if (errno) {} // expected-warning{{undefined}}
  } else {
    clang_analyzer_eval(Ret == -1); // expected-warning {{TRUE}}
    clang_analyzer_eval(errno != 0); // expected-warning {{TRUE}}
  }
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void check_fgetpos(FILE *F) {
  errno = 0;
  fpos_t Pos;
  int Ret = fgetpos(F, &Pos);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret)
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{FALSE}}
  if (errno) {} // no-warning
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void check_fsetpos(FILE *F) {
  errno = 0;
  fpos_t Pos;
  int Ret = fsetpos(F, &Pos);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret)
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{FALSE}}
  if (errno) {} // no-warning
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void check_ftell(FILE *F) {
  errno = 0;
  long Ret = ftell(F);
  clang_analyzer_eval(F != NULL); // expected-warning {{TRUE}}
  if (Ret == -1) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{FALSE}}
    clang_analyzer_eval(Ret >= 0); // expected-warning{{TRUE}}
  }
  if (errno) {} // no-warning
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void test_rewind(FILE *F) {
  errno = 0;
  rewind(F);
  clang_analyzer_eval(F != NULL); // expected-warning{{TRUE}}
  clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}
  rewind(F);
}

void test_fflush(FILE *F) {
  errno = 0;
  int Ret = fflush(F);
  clang_analyzer_eval(F != NULL); // expected-warning{{TRUE}}
                                  // expected-warning@-1{{FALSE}}
  if (Ret == EOF) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(Ret == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                     // expected-warning@-1{{FALSE}}
  }
}

void test_feof(FILE *F) {
  errno = 0;
  feof(F);
  clang_analyzer_eval(F != NULL); // expected-warning{{TRUE}}
  if (errno) {} // no-warning
  clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}
}

void test_ferror(FILE *F) {
  errno = 0;
  ferror(F);
  clang_analyzer_eval(F != NULL); // expected-warning{{TRUE}}
  if (errno) {} // no-warning
  clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}
}

void test_clearerr(FILE *F) {
  errno = 0;
  clearerr(F);
  clang_analyzer_eval(F != NULL); // expected-warning{{TRUE}}
  if (errno) {} // no-warning
  clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
                                   // expected-warning@-1{{FALSE}}
}

void freadwrite_zerosize(FILE *F) {
  fwrite(WBuf, 1, 0, F);
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
  fwrite(WBuf, 0, 1, F);
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
  fread(RBuf, 1, 0, F);
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
  fread(RBuf, 0, 1, F);
  clang_analyzer_eval(feof(F)); // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{UNKNOWN}}
}

void freadwrite_zerosize_errno(FILE *F, int A) {
  switch (A) {
  case 1:
    fwrite(WBuf, 1, 0, F);
    if (errno) {} // expected-warning{{undefined}}
    break;
  case 2:
    fwrite(WBuf, 0, 1, F);
    if (errno) {} // expected-warning{{undefined}}
    break;
  case 3:
    fread(RBuf, 1, 0, F);
    if (errno) {} // expected-warning{{undefined}}
    break;
  case 4:
    fread(RBuf, 0, 1, F);
    if (errno) {} // expected-warning{{undefined}}
    break;
  }
}
