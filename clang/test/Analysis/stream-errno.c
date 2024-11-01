// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.Stream,alpha.unix.Errno,alpha.unix.StdCLibraryFunctions,debug.ExprInspection \
// RUN:   -analyzer-config alpha.unix.StdCLibraryFunctions:ModelPOSIX=true -verify %s

#include "Inputs/system-header-simulator.h"
#include "Inputs/errno_func.h"

extern void clang_analyzer_eval(int);
extern void clang_analyzer_dump(int);
extern void clang_analyzer_printState();

void check_fopen(void) {
  FILE *F = fopen("xxx", "r");
  if (!F) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [alpha.unix.Errno]}}
}

void check_tmpfile(void) {
  FILE *F = tmpfile();
  if (!F) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno' [alpha.unix.Errno]}}
}

void check_freopen(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  F = freopen("xxx", "w", F);
  if (!F) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fclose(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fclose(F);
  if (Ret == EOF) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fread_size0(void) {
  char Buf[10];
  FILE *F = tmpfile();
  if (!F)
    return;
  fread(Buf, 0, 1, F);
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fread_nmemb0(void) {
  char Buf[10];
  FILE *F = tmpfile();
  if (!F)
    return;
  fread(Buf, 1, 0, F);
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fread(void) {
  char Buf[10];
  FILE *F = tmpfile();
  if (!F)
    return;

  int R = fread(Buf, 1, 10, F);
  if (R < 10) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    fclose(F);
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fwrite_size0(void) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  if (!F)
    return;
  fwrite(Buf, 0, 1, F);
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fwrite_nmemb0(void) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  if (!F)
    return;
  fwrite(Buf, 1, 0, F);
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fwrite(void) {
  char Buf[] = "0123456789";
  FILE *F = tmpfile();
  if (!F)
    return;

  int R = fwrite(Buf, 1, 10, F);
  if (R < 10) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    fclose(F);
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_fseek(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int S = fseek(F, 11, SEEK_SET);
  if (S != 0) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    fclose(F);
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}

void check_no_errno_change(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  errno = 1;
  clearerr(F);
  if (errno) {} // no-warning
  feof(F);
  if (errno) {} // no-warning
  ferror(F);
  if (errno) {} // no-warning
  clang_analyzer_eval(errno == 1); // expected-warning{{TRUE}}
  fclose(F);
}

void check_fgetpos(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  errno = 0;
  fpos_t Pos;
  int Ret = fgetpos(F, &Pos);
  if (Ret)
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
  if (errno) {} // no-warning
  fclose(F);
}

void check_fsetpos(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  errno = 0;
  fpos_t Pos;
  int Ret = fsetpos(F, &Pos);
  if (Ret)
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  else
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
  if (errno) {} // no-warning
  fclose(F);
}

void check_ftell(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  errno = 0;
  long Ret = ftell(F);
  if (Ret == -1) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
  } else {
    clang_analyzer_eval(errno == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(Ret >= 0); // expected-warning{{TRUE}}
  }
  if (errno) {} // no-warning
  fclose(F);
}

void check_rewind(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  errno = 0;
  rewind(F);
  clang_analyzer_eval(errno == 0);
  // expected-warning@-1{{FALSE}}
  // expected-warning@-2{{TRUE}}
  fclose(F);
}

void check_fileno(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int N = fileno(F);
  if (N == -1) {
    clang_analyzer_eval(errno != 0); // expected-warning{{TRUE}}
    if (errno) {} // no-warning
    fclose(F);
    return;
  }
  if (errno) {} // expected-warning{{An undefined value may be read from 'errno'}}
}
