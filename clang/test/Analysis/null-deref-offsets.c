// RUN: %clang_analyze_cc1 -w -triple i386-apple-darwin10 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

struct S {
  int x, y;
  int z[2];
};

void testOffsets(struct S *s, int coin) {
  if (s != 0)
    return;

  // FIXME: Here we are testing the hack that computes offsets to null pointers
  // as 0 in order to find null dereferences of not-exactly-null pointers,
  // such as &(s->y) below, which is equal to 4 rather than 0 in run-time.

  // These are indeed null.
  clang_analyzer_eval(s == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&(s->x) == 0); // expected-warning{{TRUE}}

  // FIXME: These should ideally be true.
  clang_analyzer_eval(&(s->y) == 4); // expected-warning{{FALSE}}
  clang_analyzer_eval(&(s->z[0]) == 8); // expected-warning{{FALSE}}
  clang_analyzer_eval(&(s->z[1]) == 12); // expected-warning{{FALSE}}

  // FIXME: These should ideally be false.
  clang_analyzer_eval(&(s->y) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&(s->z[0]) == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(&(s->z[1]) == 0); // expected-warning{{TRUE}}

  // But these should still be reported as null dereferences.
  if (coin)
    s->y = 5; // expected-warning{{Access to field 'y' results in a dereference of a null pointer (loaded from variable 's')}}
  else
    s->z[1] = 6; // expected-warning{{Array access (via field 'z') results in a null pointer dereference}}
}
