// RUN: %clang_analyze_cc1 -analyzer-checker=core.builtin -verify -Wno-unused -Wno-error=incompatible-pointer-types %s

typedef union {
  int y;
} U;

typedef struct { int x; } A;

void foo(void) {
  U u = {};
  A *a = &u; // expected-warning{{incompatible pointer types}}
  a->x;      // no-crash
}
