// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple i686-unknown-linux -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple armv7-a15-linux -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s
// RUN: %clang_analyze_cc1 -triple thumbv7-a15-linux -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);

typedef struct FILE FILE;
// Unorthodox EOF value.
#define EOF (-2)

int getc(FILE *);
void test_getc(FILE *fp) {

  int x;
  while ((x = getc(fp)) != EOF) {
    clang_analyzer_eval(x > 255); // expected-warning{{FALSE}}
    clang_analyzer_eval(x >= 0); // expected-warning{{TRUE}}
  }

  int y = getc(fp);
  if (y < 0) {
    clang_analyzer_eval(y == -2); // expected-warning{{TRUE}}
  }
}
