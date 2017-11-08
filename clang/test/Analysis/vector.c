// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

typedef int __attribute__((ext_vector_type(2))) V;

void clang_analyzer_numTimesReached();
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
