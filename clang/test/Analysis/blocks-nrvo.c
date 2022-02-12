// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -fblocks -verify %s

// expected-no-diagnostics

typedef struct {
  int x;
} S;

void foo(void) {
  ^{
    S s;
    return s; // no-crash
  };
}
