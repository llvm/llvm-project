// RUN: %clang_analyze_cc1 -analyzer-checker=core.NullDereference %s
// expected-no-diagnostics

typedef unsigned long size_t;

struct Point {
  int x;
  int y;
};

#define MY_OFFSETOF(T, m) ((size_t)(&((T*)0)->m))

size_t get_y_offset(void) {
  return MY_OFFSETOF(struct Point, y);
}

struct Outer {
  int pad;
  struct Point pt;
};

size_t get_nested_offset(void) {
  return MY_OFFSETOF(struct Outer, pt.x);
}
