// RUN: %clang_analyze_cc1 -verify %s -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-checker=core,debug.ExprInspection

void clang_analyzer_eval(bool);

void element_constant() {
  char arr[10];
  clang_analyzer_eval(arr + 1 > arr); // expected-warning{{TRUE}}
}

void element_known() {
  char arr[10];
  int off = 1;
  clang_analyzer_eval(arr + off > arr); // expected-warning{{TRUE}}
}

void element_constrained(int off) {
  char arr[10];
  if (off == 1) {
    clang_analyzer_eval(arr + off > arr); // expected-warning{{TRUE}}
  }
}

void element_unknown(int off) {
  char arr[10];
  clang_analyzer_eval(arr + off > arr); // expected-warning{{UNKNOWN}}
}

void element_complex(int off) {
  char arr[10];
  int comp = off * 2;
  if (off == 1) {
    clang_analyzer_eval(arr + comp); // expected-warning{{TRUE}}
  }
}

void base_constant(int *arr) {
  clang_analyzer_eval(arr + 1 > arr); // expected-warning{{TRUE}}
}

void base_known(int *arr) {
  int off = 1;
  clang_analyzer_eval(arr + off > arr); // expected-warning{{TRUE}}
}

void base_constrained(int *arr, int off) {
  if (off == 1) {
    clang_analyzer_eval(arr + off > arr); // expected-warning{{TRUE}}
  }
}

void base_unknown(int *arr, int off) {
  clang_analyzer_eval(arr + off > arr); // expected-warning{{UNKNOWN}}
}

void base_complex(int *arr, int off) {
  int comp = off * 2;
  if (off == 1) {
    clang_analyzer_eval(arr + comp > arr); // expected-warning{{TRUE}}
  }
}
