// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void symbolic_offset_pointer_cast_preserves_nonnull(_Bool i) {
  unsigned char a[2];
  unsigned char *q = a + i;
  char *r = (char *)q;

  clang_analyzer_eval(q != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(r != 0); // expected-warning{{TRUE}}

  if (!r)
    *r = 0; // no-warning
}

void differently_sized_symbolic_offset_pointer_cast_preserves_nonnull(_Bool i) {
  int a[2];
  int *q = a + i;
  char *r = (char *)q;
  int *s = (int *)r;

  clang_analyzer_eval(r != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(s == q); // expected-warning{{TRUE}}

  if (!r)
    *r = 0; // no-warning
}

void symbolic_base_pointer_cast_preserves_nonnull(int *p, _Bool i) {
  if (!p)
    return;

  int *q = p + i;
  char *r = (char *)q;

  clang_analyzer_eval(q != 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(r != 0); // expected-warning{{TRUE}}

  if (!r)
    *r = 0; // no-warning
}
