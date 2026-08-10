// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

typedef int __attribute__((ext_vector_type(2))) V;

void clang_analyzer_warnIfReached(void);
void clang_analyzer_numTimesReached(void);
void clang_analyzer_eval(int);

int flag;

V pass_through_and_set_flag(V v) {
  flag = 1;
  return v;
}

V dont_crash_and_dont_split_state(V x, V y) {
  flag = 0;
  V z = x && pass_through_and_set_flag(y);
  clang_analyzer_eval(flag); // expected-warning{{TRUE}}
  // FIXME: For now we treat vector operator && as short-circuit,
  // but in fact it is not. It should always evaluate
  // pass_through_and_set_flag(). It should not split state.
  // Now we also get FALSE on the other path.
  // expected-warning@-5{{FALSE}}

  // FIXME: Should be 1 since we should not split state.
  clang_analyzer_numTimesReached(); // expected-warning{{2}}
  return z;
}

void test_read(void) {
  V x;
  x[0] = 0;
  x[1] = 1;

  clang_analyzer_eval(x[0] == 0); // expected-warning{{TRUE}}
}

V return_vector(void) {
  V z;
  z[0] = 0;
  z[1] = 0;
  return z;
}

int test_vector_access(void) {
  return return_vector()[0]; // no-crash no-warning
}

@interface I
@property V v;
@end

// Do not crash on subscript operations into ObjC properties.
int myfunc(I *i2) {
  int out = i2.v[0]; // no-crash no-warning

  // Check that the analysis continues.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  return out;
}
