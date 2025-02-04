// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 \
// RUN:  -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached(void);

typedef struct {
  int value;
} Struct;

typedef union {
  Struct *ptr;
  long num;
} __attribute__((transparent_union)) Alias;

void foo(Struct *x);
void foo(Alias y) {
  if (y.ptr == 0) {
    // no-crash
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void foobar(long z);
void foobar(Alias z) {
  if (z.num != 42) {
    // no-crash
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void foobaz(Alias x) {
  if (x.ptr == 0) {
    // no-crash
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void bar(Struct arg) {
  foo(&arg);
  foobar(42);
  foobaz(&arg);
}
