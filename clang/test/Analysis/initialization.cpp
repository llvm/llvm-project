// RUN: %clang_cc1 -std=c++14 -triple i386-apple-darwin10 -analyze -analyzer-config eagerly-assume=false -analyzer-checker=core.uninitialized.Assign,core.builtin,debug.ExprInspection,core.uninitialized.UndefReturn -verify %s

void clang_analyzer_eval(int);

struct S {
  int a = 3;
};
S const sarr[2] = {};
void definit() {
  int i = 1;
  // FIXME: Should recognize that it is 3.
  clang_analyzer_eval(sarr[i].a); // expected-warning{{UNKNOWN}}
}

int const arr[2][2] = {};
void arr2init() {
  int i = 1;
  // FIXME: Should recognize that it is 0.
  clang_analyzer_eval(arr[i][0]); // expected-warning{{UNKNOWN}}
}

int const glob_arr1[3] = {};
void glob_array_index1() {
  clang_analyzer_eval(glob_arr1[0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[2] == 0); // expected-warning{{TRUE}}
}

void glob_invalid_index1() {
  const int *ptr = glob_arr1;
  int idx = -42;
  auto x = ptr[idx]; // expected-warning{{garbage or undefined}}
}

int const glob_arr2[4] = {1, 2};
void glob_ptr_index1() {
  int const *ptr = glob_arr2;
  clang_analyzer_eval(ptr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[2] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[3] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[4] == 0); // expected-warning{{UNDEFINED}}
}

void glob_invalid_index2() {
  const int *ptr = glob_arr2;
  int idx = 42;
  auto x = ptr[idx]; // expected-warning{{garbage or undefined}}
}

const float glob_arr3[] = {
    0.0000, 0.0235, 0.0470, 0.0706, 0.0941, 0.1176};
float no_warn_garbage_value() {
  return glob_arr3[0]; // no-warning (garbage or undefined)
}

// TODO: Support multidimensional array.
int const glob_arr4[4][2] = {};
void glob_array_index2() {
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr4[1][0] == 0); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr4[1][1] == 0); // expected-warning{{UNKNOWN}}
}

// TODO: Support multidimensional array.
void glob_invalid_index3() {
  int idx = -42;
  // FIXME: Should warn {{garbage or undefined}}.
  auto x = glob_arr4[1][idx]; // no-warning
}

// TODO: Support multidimensional array.
void glob_invalid_index4() {
  const int *ptr = glob_arr4[1];
  int idx = -42;
  // FIXME: Should warn {{garbage or undefined}}.
  auto x = ptr[idx]; // no-warning
}

// TODO: Support multidimensional array.
int const glob_arr5[4][2] = {{1}, 3, 4, 5};
void glob_array_index3() {
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[0][0] == 1); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[0][1] == 0); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[1][0] == 3); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[1][1] == 4); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[2][0] == 5); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[2][1] == 0); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[3][0] == 0); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(glob_arr5[3][1] == 0); // expected-warning{{UNKNOWN}}
}

// TODO: Support multidimensional array.
void glob_ptr_index2() {
  int const *ptr = glob_arr5[1];
  // FIXME: Should be TRUE.
  clang_analyzer_eval(ptr[0] == 3); // expected-warning{{UNKNOWN}}
  // FIXME: Should be TRUE.
  clang_analyzer_eval(ptr[1] == 4); // expected-warning{{UNKNOWN}}
  // FIXME: Should be UNDEFINED.
  clang_analyzer_eval(ptr[2] == 5); // expected-warning{{UNKNOWN}}
  // FIXME: Should be UNDEFINED.
  clang_analyzer_eval(ptr[3] == 0); // expected-warning{{UNKNOWN}}
  // FIXME: Should be UNDEFINED.
  clang_analyzer_eval(ptr[4] == 0); // expected-warning{{UNKNOWN}}
}

// TODO: Support multidimensional array.
void glob_invalid_index5() {
  int idx = -42;
  // FIXME: Should warn {{garbage or undefined}}.
  auto x = glob_arr5[1][idx]; // no-warning
}

// TODO: Support multidimensional array.
void glob_invalid_index6() {
  int const *ptr = &glob_arr5[1][0];
  int idx = 42;
  // FIXME: Should warn {{garbage or undefined}}.
  auto x = ptr[idx]; // // no-warning
}
