// RUN: %clang_analyze_cc1 -verify %s \
// RUN: -analyzer-checker=core \
// RUN: -analyzer-checker=unix.Stream \
// RUN: -analyzer-config unix.Stream:Pedantic=true \
// RUN: -analyzer-checker=debug.StreamTester \
// RUN: -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);
void clang_analyzer_warnIfReached(void);
void StreamTesterChecker_make_feof_stream(FILE *);
void StreamTesterChecker_make_ferror_stream(FILE *);
void StreamTesterChecker_make_ferror_indeterminate_stream(FILE *);

void error_fopen(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  clang_analyzer_eval(feof(F)); // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  fclose(F);
}

void error_fdopen(int fd) {
  FILE *F = fdopen(fd, "r");
  if (!F)
    return;
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
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
  StreamTesterChecker_make_ferror_indeterminate_stream(F);
  clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
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
  StreamTesterChecker_make_ferror_indeterminate_stream(F);
  clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
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
  Ret = fread(Buf, 1, 10, F); // expected-warning {{Use of a stream that might be already closed}}
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
  Ret = fwrite(0, 1, 10, F); // expected-warning {{Use of a stream that might be already closed}}
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
  fgetc(F); // expected-warning {{Use of a stream that might be already closed}}
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
  fgets(Buf, sizeof(Buf), F);         // expected-warning {{Use of a stream that might be already closed}}
}

void error_fputc(int fd) {
  FILE *F = fdopen(fd, "w");
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
  fputc('A', F); // expected-warning {{Use of a stream that might be already closed}}
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
  fputs("ABC", F); // expected-warning {{Use of a stream that might be already closed}}
}

void error_fprintf(void) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fprintf(F, "aaa");
  if (Ret >= 0) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
    fprintf(F, "bbb");                         // no-warning
  } else {
    clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    fprintf(F, "bbb");              // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
  fprintf(F, "ccc"); // expected-warning {{Use of a stream that might be already closed}}
}

void error_fscanf(int *A) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = fscanf(F, "a%ib", A);
  if (Ret >= 0) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
    fscanf(F, "bbb");                          // no-warning
  } else {
    if (ferror(F)) {
      clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
      fscanf(F, "bbb");               // expected-warning {{might be 'indeterminate'}}
    } else if (feof(F)) {
      clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
      fscanf(F, "bbb");               // expected-warning {{is in EOF state}}
      clang_analyzer_eval(feof(F));   // expected-warning {{TRUE}}
    } else {
      clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
      fscanf(F, "bbb");               // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  fscanf(F, "ccc"); // expected-warning {{Use of a stream that might be already closed}}
}

void error_ungetc(int TestIndeterminate) {
  FILE *F = tmpfile();
  if (!F)
    return;
  int Ret = ungetc('X', F);
  clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  if (Ret == EOF) {
    clang_analyzer_warnIfReached();          // expected-warning {{REACHABLE}}
  } else {
    clang_analyzer_eval(Ret == 'X');         // expected-warning {{TRUE}}
  }
  fputc('Y', F);                             // no-warning
  if (TestIndeterminate) {
    StreamTesterChecker_make_ferror_indeterminate_stream(F);
    ungetc('X', F);                          // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
  ungetc('A', F);                            // expected-warning {{Use of a stream that might be already closed}}
}

void error_getdelim(char *P, size_t Sz) {
  FILE *F = tmpfile();
  if (!F)
    return;
  ssize_t Ret = getdelim(&P, &Sz, '\t', F);
  if (Ret >= 0) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(Ret == -1);            // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{TRUE}}
    if (feof(F)) {
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
      getdelim(&P, &Sz, '\n', F);     // expected-warning {{Read function called when stream is in EOF state}}
    } else {
      clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
      getdelim(&P, &Sz, '\n', F);     // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  getdelim(&P, &Sz, '\n', F);         // expected-warning {{Use of a stream that might be already closed}}
}

void error_getline(char *P, size_t Sz) {
  FILE *F = tmpfile();
  if (!F)
    return;
  ssize_t Ret = getline(&P, &Sz, F);
  if (Ret >= 0) {
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(Ret == -1);            // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F) || ferror(F)); // expected-warning {{TRUE}}
    if (feof(F)) {
      clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
      getline(&P, &Sz, F);            // expected-warning {{Read function called when stream is in EOF state}}
    } else {
      clang_analyzer_eval(ferror(F)); // expected-warning {{TRUE}}
      getline(&P, &Sz, F);            // expected-warning {{might be 'indeterminate'}}
    }
  }
  fclose(F);
  getline(&P, &Sz, F);                // expected-warning {{Use of a stream that might be already closed}}
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
    clang_analyzer_eval(rc == -1);     // expected-warning {{TRUE}}
    int IsFEof = feof(F), IsFError = ferror(F);
    // Get ferror or no error.
    clang_analyzer_eval(IsFError);     // expected-warning {{FALSE}} \
                                       // expected-warning {{TRUE}}
    clang_analyzer_eval(IsFEof);       // expected-warning {{FALSE}}
    // Error flags should not change.
    clang_analyzer_eval(feof(F));      // expected-warning {{FALSE}}
    if (IsFError)
      clang_analyzer_eval(ferror(F));  // expected-warning {{TRUE}}
  } else {
    clang_analyzer_eval(feof(F));      // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F));    // expected-warning {{FALSE}}
    // Error flags should not change.
    clang_analyzer_eval(feof(F));      // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F));    // expected-warning {{FALSE}}
  }
  fclose(F);
}

void error_fseeko(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseeko(F, 1, SEEK_SET);
  if (rc) {
    // Get ferror or no error.
    clang_analyzer_eval(ferror(F));  // expected-warning {{FALSE}} \
                                     // expected-warning {{TRUE}}
    clang_analyzer_eval(feof(F));    // expected-warning {{FALSE}}
  } else {
    clang_analyzer_eval(feof(F));    // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F));  // expected-warning {{FALSE}}
  }
  fclose(F);
}

void error_fseek_0(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseek(F, 0, SEEK_SET);
  if (rc == -1) {
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

void error_fseeko_0(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int rc = fseeko(F, 0, SEEK_SET);
  if (rc) {
    int IsFEof = feof(F), IsFError = ferror(F);
    // Get ferror or no error, but not feof.
    clang_analyzer_eval(IsFError);
    // expected-warning@-1 {{FALSE}}
    // expected-warning@-2 {{TRUE}}
    clang_analyzer_eval(IsFEof);
    // expected-warning@-1 {{FALSE}}
  } else {
    clang_analyzer_eval(feof(F));   // expected-warning {{FALSE}}
    clang_analyzer_eval(ferror(F)); // expected-warning {{FALSE}}
  }
  fclose(F);
}

void error_ftell(int TestIndeterminate) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  long rc = ftell(F);
  if (rc >= 0)
    clang_analyzer_warnIfReached();          // expected-warning {{REACHABLE}}
  else
    clang_analyzer_eval(rc == -1);           // expected-warning {{TRUE}}
  clang_analyzer_eval(feof(F) && ferror(F)); // expected-warning {{FALSE}}
  StreamTesterChecker_make_feof_stream(F);
  rc = ftell(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{TRUE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{FALSE}}
  StreamTesterChecker_make_ferror_stream(F);
  rc = ftell(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{TRUE}}
  if (TestIndeterminate) {
    StreamTesterChecker_make_ferror_indeterminate_stream(F);
    ftell(F);                                // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
}

void error_ftello(int TestIndeterminate) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  off_t rc = ftello(F);
  if (rc >= 0)
    clang_analyzer_warnIfReached();          // expected-warning {{REACHABLE}}
  else
    clang_analyzer_eval(rc == -1);           // expected-warning {{TRUE}}
  clang_analyzer_eval(feof(F) && ferror(F)); // expected-warning {{FALSE}}
  StreamTesterChecker_make_feof_stream(F);
  rc = ftello(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{TRUE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{FALSE}}
  StreamTesterChecker_make_ferror_stream(F);
  rc = ftello(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{TRUE}}
  if (TestIndeterminate) {
    StreamTesterChecker_make_ferror_indeterminate_stream(F);
    ftell(F);                                // expected-warning {{might be 'indeterminate'}}
  }
  fclose(F);
}

void error_fileno(void) {
  FILE *F = fopen("file", "r");
  if (!F)
    return;
  int N = fileno(F);
  clang_analyzer_eval(N >= 0); // expected-warning {{TRUE}}
  clang_analyzer_eval(feof(F) && ferror(F)); // expected-warning {{FALSE}}
  StreamTesterChecker_make_feof_stream(F);
  N = fileno(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{TRUE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{FALSE}}
  StreamTesterChecker_make_ferror_stream(F);
  N = fileno(F);
  clang_analyzer_eval(feof(F));              // expected-warning {{FALSE}}
  clang_analyzer_eval(ferror(F));            // expected-warning {{TRUE}}
  StreamTesterChecker_make_ferror_indeterminate_stream(F);
  fileno(F);                                 // no warning
  fclose(F);
}

void error_fflush_on_non_null_stream_clear_error_states(void) {
  FILE *F0 = tmpfile(), *F1 = tmpfile();
  // `fflush` clears a non-EOF stream's error state.
  if (F0) {
    StreamTesterChecker_make_ferror_stream(F0);
    if (fflush(F0) == 0) {             // no-warning
      clang_analyzer_eval(ferror(F0)); // expected-warning {{FALSE}}
      clang_analyzer_eval(feof(F0));   // expected-warning {{FALSE}}
    }
    fclose(F0);
  }
  // `fflush` clears an EOF stream's error state.
  if (F1) {
    StreamTesterChecker_make_feof_stream(F1);
    if (fflush(F1) == 0) {             // no-warning
      clang_analyzer_eval(ferror(F1)); // expected-warning {{FALSE}}
      clang_analyzer_eval(feof(F1));   // expected-warning {{TRUE}}
    }
    fclose(F1);
  }
}

void error_fflush_on_null_stream_clear_error_states(void) {
  FILE *F0 = tmpfile(), *F1 = tmpfile();
  // `fflush` clears all stream's error states, while retains their EOF states.
  if (F0 && F1) {
    StreamTesterChecker_make_ferror_stream(F0);
    StreamTesterChecker_make_feof_stream(F1);
    if (fflush(NULL) == 0) {           // no-warning
      clang_analyzer_eval(ferror(F0)); // expected-warning {{FALSE}}
      clang_analyzer_eval(feof(F0));   // expected-warning {{FALSE}}
      clang_analyzer_eval(ferror(F1)); // expected-warning {{FALSE}}
      clang_analyzer_eval(feof(F1));   // expected-warning {{TRUE}}
    }
  }
  if (F0)
    fclose(F0);
  if (F1)
    fclose(F1);
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
