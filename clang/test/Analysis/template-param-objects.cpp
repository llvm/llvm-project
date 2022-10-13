// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:    -analyzer-config eagerly-assume=false -std=c++20 -verify %s

template <class T> void clang_analyzer_dump(T);
void clang_analyzer_eval(bool);

struct Box {
  int value;
};
bool operator ==(Box lhs, Box rhs) {
  return lhs.value == rhs.value;
}
template <Box V> void dumps() {
  clang_analyzer_dump(V);        // expected-warning {{lazyCompoundVal}}
  clang_analyzer_dump(&V);       // expected-warning {{Unknown}}
  clang_analyzer_dump(V.value);  // expected-warning {{Unknown}} FIXME: It should be '6 S32b'.
  clang_analyzer_dump(&V.value); // expected-warning {{Unknown}}
}
template void dumps<Box{6}>();

// [temp.param].7.3.2:
// "All such template parameters in the program of the same type with the
// same value denote the same template parameter object."
template <Box A1, Box A2, Box B1, Box B2> void stable_addresses() {
  clang_analyzer_eval(&A1 == &A2); // expected-warning {{UNKNOWN}} FIXME: It should be TRUE.
  clang_analyzer_eval(&B1 == &B2); // expected-warning {{UNKNOWN}} FIXME: It should be TRUE.
  clang_analyzer_eval(&A1 == &B2); // expected-warning {{UNKNOWN}} FIXME: It should be FALSE.

  clang_analyzer_eval(A1 == A2); // expected-warning {{UNKNOWN}} FIXME: It should be TRUE.
  clang_analyzer_eval(B1 == B2); // expected-warning {{UNKNOWN}} FIXME: It should be TRUE.
  clang_analyzer_eval(A1 == B2); // expected-warning {{UNKNOWN}} FIXME: It should be FALSE.
}
template void stable_addresses<Box{1}, Box{1}, Box{2}, Box{2}>();
