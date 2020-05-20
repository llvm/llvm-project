// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=alpha.unix.Stream \
// RUN: -analyzer-checker=debug.StreamTester \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();
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

void error_fread() {
  FILE *F = tmpfile();
  if (!F)
    return;
  char Buf[10];
  int Ret = fread(Buf, 1, 10, F);
  if (Ret == 10) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{TRUE}}
    if (feof(F)) {
      clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
      fread(Buf, 1, 10, F);           // expected-warning {{Read function called when stream is in EOF state}}
      clang_analyzer_eval(feof(F));   // expected-warning {{TRUE}}
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
    }
    if (ferror(F)) {
      clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
      fread(Buf, 1, 10, F);           // no warning
    }
  }
  fclose(F);
  Ret = fread(Buf, 1, 10, F); // expected-warning {{Stream might be already closed}}
}

void error_fwrite() {
  FILE *F = tmpfile();
  if (!F)
    return;
  const char *Buf = "123456789";
  int Ret = fwrite(Buf, 1, 10, F);
  if (Ret == 10) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
    fwrite(0, 1, 10, F);            // no warning
  }
  fclose(F);
  Ret = fwrite(0, 1, 10, F); // expected-warning {{Stream might be already closed}}
}

void freadwrite_zerosize(FILE *F) {
  fwrite(0, 1, 0, F);
  fwrite(0, 0, 1, F);
  fread(0, 1, 0, F);
  fread(0, 0, 1, F);
}

void freadwrite_zerosize_eofstate(FILE *F) {
  fwrite(0, 1, 0, F);
  fwrite(0, 0, 1, F);
  fread(0, 1, 0, F); // expected-warning {{Read function called when stream is in EOF state}}
  fread(0, 0, 1, F); // expected-warning {{Read function called when stream is in EOF state}}
}

void error_fread_fwrite_zerosize() {
  FILE *F = fopen("file", "r");
  if (!F)
    return;

  freadwrite_zerosize(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}

  StreamTesterChecker_make_ferror_stream(F);
  freadwrite_zerosize(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}

  StreamTesterChecker_make_feof_stream(F);
  freadwrite_zerosize_eofstate(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{TRUE}}
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
