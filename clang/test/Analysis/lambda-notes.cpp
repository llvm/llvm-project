// RUN: %clang_analyze_cc1 -std=c++11 -analyzer-checker=core.DivideZero -analyzer-config inline-lambdas=true -analyzer-output plist -verify %s -o %t
// RUN: tail -n +11 %t | %diff_plist %S/Inputs/expected-plists/lambda-notes.cpp.plist -


// Diagnostic inside a lambda

void diagnosticFromLambda() {
  int i = 0;
  [=] {
    int p = 5/i; // expected-warning{{Division by zero}}
    (void)p;
  }();
}
