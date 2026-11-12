// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

// Test parameter 'a' is registered to LiveVariables analysis data although it
// is not referenced in the function body. 
// Before processing 'return 1;', in RemoveDeadBindings(), we query the liveness
// of 'a', because we have a binding for it due to parameter passing.
int f1(int a) {
  return 1;
}

void f2(void) {
  int x;
  x = f1(1);
}
