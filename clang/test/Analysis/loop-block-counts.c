// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void callee(void **p) {
  int x;
  *p = &x;
  // expected-warning@-1 {{Address of stack memory associated with local \
variable 'x' is still referred to by the stack variable 'arr' upon \
returning to the caller}}
}

void loop() {
  void *arr[2];
  for (int i = 0; i < 2; ++i)
    callee(&arr[i]);
  // FIXME: Should be UNKNOWN.
  clang_analyzer_eval(arr[0] == arr[1]); // expected-warning{{FALSE}}
}

void loopWithCall() {
  void *arr[2];
  for (int i = 0; i < 2; ++i) {
    int x;
    arr[i] = &x;
  }
  // FIXME: Should be UNKNOWN.
  clang_analyzer_eval(arr[0] == arr[1]); // expected-warning{{TRUE}}
}
