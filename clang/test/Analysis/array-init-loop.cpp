// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

void array_init() {
  int arr[] = {1, 2, 3, 4, 5};

  auto [a, b, c, d, e] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(d == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(e == 5); // expected-warning{{TRUE}}
}

void array_uninit() {
  int arr[5];

  auto [a, b, c, d, e] = arr;

  int x = e; // expected-warning{{Assigned value is garbage or undefined}}
}

void lambda_init() {
  int arr[] = {1, 2, 3, 4, 5};

  auto l = [arr] { return arr[0]; }();
  clang_analyzer_eval(l == 1); // expected-warning{{TRUE}}

  l = [arr] { return arr[1]; }();
  clang_analyzer_eval(l == 2); // expected-warning{{TRUE}}

  l = [arr] { return arr[2]; }();
  clang_analyzer_eval(l == 3); // expected-warning{{TRUE}}

  l = [arr] { return arr[3]; }();
  clang_analyzer_eval(l == 4); // expected-warning{{TRUE}}

  l = [arr] { return arr[4]; }();
  clang_analyzer_eval(l == 5); // expected-warning{{TRUE}}
}

void lambda_uninit() {
  int arr[5];

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal
  int l = [arr] { return arr[0]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[1]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[2]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[3]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}

  l = [arr] { return arr[4]; }();
  clang_analyzer_eval(l); // expected-warning{{UNKNOWN}}
}

struct S {
  int arr[5];
};

void copy_ctor_init() {
  S orig;
  orig.arr[0] = 1;
  orig.arr[1] = 2;
  orig.arr[2] = 3;
  orig.arr[3] = 4;
  orig.arr[4] = 5;

  S copy = orig;
  clang_analyzer_eval(copy.arr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(copy.arr[4] == 5); // expected-warning{{TRUE}}
}

void copy_ctor_uninit() {
  S orig;

  S copy = orig;

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal.
  // If the struct is not considered a small struct, instead of a copy, we store a lazy compound value.
  // As the struct has an array data member, it is not considered small.
  clang_analyzer_eval(copy.arr[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[1]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[3]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(copy.arr[4]); // expected-warning{{UNKNOWN}}
}

void move_ctor_init() {
  S orig;
  orig.arr[0] = 1;
  orig.arr[1] = 2;
  orig.arr[2] = 3;
  orig.arr[3] = 4;
  orig.arr[4] = 5;

  S moved = (S &&) orig;

  clang_analyzer_eval(moved.arr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(moved.arr[4] == 5); // expected-warning{{TRUE}}
}

void move_ctor_uninit() {
  S orig;

  S moved = (S &&) orig;

  // FIXME: These should be Undefined, but we fail to read Undefined from a lazyCompoundVal.
  clang_analyzer_eval(moved.arr[0]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[1]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[2]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[3]); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(moved.arr[4]); // expected-warning{{UNKNOWN}}
}
