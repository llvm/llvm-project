// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=alpha.unix.Stream \
// RUN: -analyzer-checker=debug.StreamTester \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);
void StreamTesterChecker_make_feof_stream(FILE *);
void StreamTesterChecker_make_ferror_stream(FILE *);

void error_fopen() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  clang_analyzer_eval(feof(F)); // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void error_freopen() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  F = freopen(0, "w", F);
  if (!F)
    return;
  clang_analyzer_eval(feof(F)); // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void stream_error_feof() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  StreamTesterChecker_make_feof_stream(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{TRUE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  clearerr(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void stream_error_ferror() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  StreamTesterChecker_make_ferror_stream(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
  clearerr(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void error_fseek() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseek(F, 0, SEEK_SET);
  if (rc) {
    int IsFEof = feof(F), IsFError = ferror(F);
    // Get feof or ferror or no error.
    clang_analyzer_eval(IsFEof || IsFError);
    // expected-warning@-1 {{FALSE}}
    // expected-warning@-2 {{TRUE}}
    clang_analyzer_eval(IsFEof && IsFError); // expected-warning {{FALSE}}
    // Error flags should not change.
    if (IsFEof)
      clang_analyzer_eval(feof(F)); // expected-warning {{TRUE}}
    else
      clang_analyzer_eval(feof(F)); // expected-warning {{FALSE}}
    if (IsFError)
      clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
    else
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
    // Error flags should not change.
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  }
  fclose(F);
}
