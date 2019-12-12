// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c99 -verify %s

void clang_analyzer_eval(int);

void array_init() {
  int a[5] = {[4] = 29, [2] = 15, [0] = 4};
  clang_analyzer_eval(a[0] == 4);  // expected-warning{{TRUE}}
  clang_analyzer_eval(a[1] == 0);  // expected-warning{{TRUE}}
  clang_analyzer_eval(a[2] == 15); // expected-warning{{TRUE}}
  clang_analyzer_eval(a[3] == 0);  // expected-warning{{TRUE}}
  clang_analyzer_eval(a[4] == 29); // expected-warning{{TRUE}}
  int b[5] = {[0 ... 2] = 1, [4] = 5};
  clang_analyzer_eval(b[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[1] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[2] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[3] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(b[4] == 5); // expected-warning{{TRUE}}
}

struct point {
  int x, y;
};

void struct_init() {
  struct point p = {.y = 5, .x = 3};
  clang_analyzer_eval(p.x == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.y == 5); // expected-warning{{TRUE}}
}

void array_of_struct() {
  struct point ptarray[3] = { [2].y = 1, [2].x = 2, [0].x = 3 };
  clang_analyzer_eval(ptarray[0].x == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptarray[0].y == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptarray[1].x == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptarray[1].y == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptarray[2].x == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptarray[2].y == 1); // expected-warning{{TRUE}}
}
