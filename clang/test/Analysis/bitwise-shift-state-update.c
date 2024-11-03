// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config eagerly-assume=false \
// RUN:    -verify=expected,c \
// RUN:    -triple x86_64-pc-linux-gnu -x c %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core.BitwiseShift \
// RUN:    -analyzer-config core.BitwiseShift:Pedantic=true \
// RUN:    -analyzer-checker=debug.ExprInspection \
// RUN:    -analyzer-config eagerly-assume=false \
// RUN:    -verify=expected,cxx \
// RUN:    -triple x86_64-pc-linux-gnu -x c++ -std=c++14 %s \
// RUN:    -Wno-shift-count-negative -Wno-shift-negative-value \
// RUN:    -Wno-shift-count-overflow -Wno-shift-overflow \
// RUN:    -Wno-shift-sign-overflow

// Tests for validating the state updates provided by the BitwiseShift checker.
// These clang_analyzer_value() tests are in a separate file because
// debug.ExprInspection repeats each 'warning' with an superfluous 'note', so
// note level output (-analyzer-output=text) is not enabled in this file.

void clang_analyzer_value(int);
void clang_analyzer_eval(int);

int state_update_generic(int left, int right) {
  int x = left << right;
  clang_analyzer_value(left); // expected-warning {{32s:{ [0, 2147483647] } }}
  clang_analyzer_value(right); // expected-warning {{32s:{ [0, 31] } }}
  return x;
}

int state_update_exact_shift(int arg) {
  int x = 65535 << arg;
  clang_analyzer_value(arg);
  // c-warning@-1 {{32s:{ [0, 15] } }}
  // cxx-warning@-2 {{32s:{ [0, 16] } }}
  return x;
}
