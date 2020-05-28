// RUN: %clang_cc1 -triple=i386-apple-darwin10 -verify %s -analyze \
// RUN:   -analyzer-checker=debug.ExprInspection

#define NULL 0
void clang_analyzer_eval(int);

// pr28449: Used to crash.
void foo(void) {
  static const unsigned short array[] = (const unsigned short[]){0x0F00};
  clang_analyzer_eval(array[0] == 0x0F00); // expected-warning{{TRUE}}
}

// check that we propagate info through compound literal regions
void bar() {
  int *integers = (int[]){1, 2, 3};
  clang_analyzer_eval(integers[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(integers[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(integers[2] == 3); // expected-warning{{TRUE}}

  int **pointers = (int *[]){&integers[0], NULL};
  clang_analyzer_eval(pointers[0] == NULL); // expected-warning{{FALSE}}
  clang_analyzer_eval(pointers[1] == NULL); // expected-warning{{TRUE}}
}
