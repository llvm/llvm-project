// Check the basic reporting/warning and the application of constraints.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify=report

// Check the bugpath related to the reports.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-output=text \
// RUN:   -verify=bugpath

void clang_analyzer_eval(int);

typedef struct FILE FILE;
typedef typeof(sizeof(int)) size_t;
size_t fread(void *restrict, size_t, size_t, FILE *restrict);
void test_notnull_concrete(FILE *fp) {
  fread(0, sizeof(int), 10, fp); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
void test_notnull_symbolic(FILE *fp, int *buf) {
  fread(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(buf != 0); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{'buf' is not equal to null}}
}
void test_notnull_symbolic2(FILE *fp, int *buf) {
  if (!buf)                          // bugpath-note{{Assuming 'buf' is null}} \
            // bugpath-note{{Taking true branch}}
    fread(buf, sizeof(int), 10, fp); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // report-note{{}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{}} \
    // bugpath-note{{Function argument constraint is not satisfied}}
}
typedef __WCHAR_TYPE__ wchar_t;
// This is one test case for the ARR38-C SEI-CERT rule.
void ARR38_C_F(FILE *file) {
  enum { BUFFER_SIZE = 1024 };
  wchar_t wbuf[BUFFER_SIZE]; // bugpath-note{{'wbuf' initialized here}}

  const size_t size = sizeof(*wbuf);   // bugpath-note{{'size' initialized to}}
  const size_t nitems = sizeof(wbuf);  // bugpath-note{{'nitems' initialized to}}

  // The 3rd parameter should be the number of elements to read, not
  // the size in bytes.
  fread(wbuf, size, nitems, file); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // report-note{{}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
}
