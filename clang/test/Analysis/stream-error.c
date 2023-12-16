// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=alpha.unix.Stream \
// RUN: -analyzer-checker=debug.StreamTester \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);
void clang_analyzer_warnIfReached(void);
void StreamTesterChecker_make_feof_stream(FILE *);
void StreamTesterChecker_make_ferror_stream(FILE *);

void error_fopen(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  clang_analyzer_eval(feof(F)); // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void error_freopen(void) {
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

void stream_error_feof(void) {
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

void stream_error_ferror(void) {
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

void error_fread(void) {
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
      fread(Buf, 1, 10, F);           // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  Ret = fread(Buf, 1, 10, F); // expected-warning {{Stream might be already closed}}
}

void error_fwrite(void) {
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
    fwrite(0, 1, 10, F);            // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
  Ret = fwrite(0, 1, 10, F); // expected-warning {{Stream might be already closed}}
}

void error_fgetc(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fgetc(F);
  if (0 <= Ret && Ret <= 255) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(Ret == EOF);           // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{TRUE}}
    if (feof(F)) {
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
      fgetc(F);                       // expected-warning {{Read function called when stream is in EOF state}}
    } else {
      clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
      fgetc(F);                       // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  fgetc(F); // expected-warning {{Stream might be already closed}}
}

void error_fgets(void) {
  FILE *F = tmpfile();
  char Buf[256];
  if (!F)
    return;
  char *Ret = fgets(Buf, sizeof(Buf), F);
  if (Ret == Buf) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(Ret == NULL);          // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{TRUE}}
    if (feof(F)) {
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
      fgets(Buf, sizeof(Buf), F);     // expected-warning {{Read function called when stream is in EOF state}}
    } else {
      clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
      fgets(Buf, sizeof(Buf), F);     // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  fgets(Buf, sizeof(Buf), F);         // expected-warning {{Stream might be already closed}}
}

void error_fputc(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fputc('X', F);
  if (Ret == EOF) {
    clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    fputc('Y', F);                  // expected-warning {{might be 'indeterminate'}}
  } else {
    clang_analyzer_eval(Ret == 'X');             // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F) || ferror(F));   // expected-warning {{FALSE}}
    fputc('Y', F);                               // no-warning
  }
  fclose(F);
  fputc('A', F); // expected-warning {{Stream might be already closed}}
}

void error_fputs(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fputs("XYZ", F);
  if (Ret >= 0) {
    clang_analyzer_eval(feof(F) || ferror(F));   // expected-warning {{FALSE}}
    fputs("QWD", F);                             // no-warning
  } else {
    clang_analyzer_eval(Ret == EOF); // expected-warning {{TRUE}}
    clang_analyzer_eval(ferror(F));  // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F));    // expected-warning {{FALSE}}
    fputs("QWD", F);                 // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
  fputs("ABC", F); // expected-warning {{Stream might be already closed}}
}

void write_after_eof_is_allowed(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  StreamTesterChecker_make_feof_stream(F);
  if (fputs("QWD", F) >= 0)                    // no-warning
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  StreamTesterChecker_make_feof_stream(F);
  if (fputc('Q', F) == 'Q')                    // no-warning
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  StreamTesterChecker_make_feof_stream(F);
  if (fwrite("012345678", 1, 10, F) == 10)     // no-warning
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void freadwrite_zerosize(FILE *F) {
  size_t Ret;
  Ret = fwrite(0, 1, 0, F);
  clang_analyzer_dump(Ret); // expected-warning {{0 }}
  Ret = fwrite(0, 0, 1, F);
  clang_analyzer_dump(Ret); // expected-warning {{0 }}
  Ret = fread(0, 1, 0, F);
  clang_analyzer_dump(Ret); // expected-warning {{0 }}
  Ret = fread(0, 0, 1, F);
  clang_analyzer_dump(Ret); // expected-warning {{0 }}
}

void freadwrite_zerosize_eofstate(FILE *F) {
  fwrite(0, 1, 0, F);
  fwrite(0, 0, 1, F);
  fread(0, 1, 0, F); // expected-warning {{Read function called when stream is in EOF state}}
  fread(0, 0, 1, F); // expected-warning {{Read function called when stream is in EOF state}}
}

void error_fread_fwrite_zerosize(void) {
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

void error_fseek(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseek(F, 1, SEEK_SET);
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

void error_fseek_0(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseek(F, 0, SEEK_SET);
  if (rc) {
    int IsFEof = feof(F), IsFError = ferror(F);
    // Get ferror or no error, but not feof.
    clang_analyzer_eval(IsFError);
    // expected-warning@-1 {{FALSE}}
    // expected-warning@-2 {{TRUE}}
    clang_analyzer_eval(IsFEof);
    // expected-warning@-1 {{FALSE}}
    // Error flags should not change.
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

void error_indeterminate(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;
  const char *Buf = "123456789";
  int rc = fseek(F, 0, SEEK_SET);
  if (rc) {
    if (feof(F)) {
      fwrite(Buf, 1, 10, F); // no warning
    } else if (ferror(F)) {
      fwrite(Buf, 1, 10, F); // expected-warning {{might be 'indeterminate'}}
    } else {
      fwrite(Buf, 1, 10, F); // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
}

void error_indeterminate_clearerr(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;
  const char *Buf = "123456789";
  int rc = fseek(F, 0, SEEK_SET);
  if (rc) {
    if (feof(F)) {
      clearerr(F);
      fwrite(Buf, 1, 10, F); // no warning
    } else if (ferror(F)) {
      clearerr(F);
      fwrite(Buf, 1, 10, F); // expected-warning {{might be 'indeterminate'}}
    } else {
      clearerr(F);
      fwrite(Buf, 1, 10, F); // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
}

void error_indeterminate_feof1(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;
  char Buf[10];
  if (fread(Buf, 1, 10, F) < 10) {
    if (feof(F)) {
      // error is feof, should be non-indeterminate
      fwrite("1", 1, 1, F); // no warning
    }
  }
  fclose(F);
}

void error_indeterminate_feof2(void) {
  FILE *F = fopen("file", "r+");
  if (!F)
    return;
  char Buf[10];
  if (fread(Buf, 1, 10, F) < 10) {
    if (ferror(F) == 0) {
      // error is feof, should be non-indeterminate
      fwrite("1", 1, 1, F); // no warning
    }
  }
  fclose(F);
}
