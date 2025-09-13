// RUN: %clang_analyze_cc1 -analyzer-checker=core.builtin -verify -Wno-unused %s
// expected-no-diagnostics

typedef union {
  int y;
} U;

typedef struct { int x; } A;

void foo(void) {
  U u = {};
  A *a = (A*)&u;
  a->x;      // no-crash
}
