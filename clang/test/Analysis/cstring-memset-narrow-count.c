// RUN: %clang_analyze_cc1 -verify %s -fno-builtin-memset \
// RUN:   -analyzer-checker=core,unix.cstring,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

void clang_analyzer_eval(int);

// Use a narrow count type to make sure sizeof is not truncated.
void *memset(void *s, int c, unsigned char n);

struct NarrowCount {
  char buf[300];
  int x;
};

void memset_narrow_count_does_not_zero_whole_object(struct NarrowCount *p) {
  memset(p, 0, sizeof(*p) & 0xff);
  clang_analyzer_eval(p->x == 0); // expected-warning{{UNKNOWN}}
}
