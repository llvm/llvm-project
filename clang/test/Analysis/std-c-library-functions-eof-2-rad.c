// RUN: %clang_analyze_cc1 -std=c23 -analyzer-checker=core,unix.StdCLibraryFunctions,debug.ExprInspection -verify -analyzer-config eagerly-assume=false %s

void clang_analyzer_eval(int);

typedef struct FILE FILE;
/// Test that the static analyzer doesn't interpret the most significant bit as the sign bit.
// Unorthodox EOF value with a power of 2 radix
#define EOF (-0b11)

int getc(FILE *);
void test_getc(FILE *fp) {
  int y = getc(fp);
  if (y < 0) {
    clang_analyzer_eval(y == EOF); // expected-warning{{TRUE}}
  }
}
