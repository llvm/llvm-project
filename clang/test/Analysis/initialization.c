// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-config eagerly-assume=false  -analyzer-checker=core.uninitialized.Assign,debug.ExprInspection -verify %s

void clang_analyzer_eval(int);

void initbug() {
  const union { float a; } u = {};
  (void)u.a; // no-crash
}

int const parr[2] = {1};
void constarr() {
  int i = 2;
  clang_analyzer_eval(parr[i]); // expected-warning{{UNDEFINED}}
  i = 1;
  clang_analyzer_eval(parr[i] == 0); // expected-warning{{TRUE}}
  i = -1;
  clang_analyzer_eval(parr[i]); // expected-warning{{UNDEFINED}}
}

struct SM {
  int a;
  int b;
};
const struct SM sm = {.a = 1};
void multinit() {
  clang_analyzer_eval(sm.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(sm.b == 0); // expected-warning{{TRUE}}
}

const int glob_arr1[6] = {[2] = 3, [0] = 1, [1] = 2, [3] = 4};
void glob_array_index1() {
  clang_analyzer_eval(glob_arr1[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[4] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr1[5] == 0); // expected-warning{{TRUE}}
}

void glob_array_index2() {
  const int *ptr = glob_arr1;
  clang_analyzer_eval(ptr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[3] == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[4] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(ptr[5] == 0); // expected-warning{{TRUE}}
}

void glob_invalid_index1() {
  int x = -42;
  int res = glob_arr1[x]; // expected-warning{{garbage or undefined}}
}

void glob_invalid_index2() {
  const int *ptr = glob_arr1;
  int x = 42;
  int res = ptr[x]; // expected-warning{{garbage or undefined}}
}

const int glob_arr2[3][3] = {[0][0] = 1, [1][1] = 5, [2][0] = 7};
void glob_arr_index3() {
  clang_analyzer_eval(glob_arr2[0][0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[0][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[0][2] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[1][0] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[1][1] == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[1][2] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[2][0] == 7); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[2][1] == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr2[2][2] == 0); // expected-warning{{TRUE}}
}

void negative_index() {
  int x = 2, y = -2;
  clang_analyzer_eval(glob_arr2[x][y] == 5); // expected-warning{{UNDEFINED}}
  x = 3;
  y = -3;
  clang_analyzer_eval(glob_arr2[x][y] == 7); // expected-warning{{UNDEFINED}}
}

void glob_invalid_index3() {
  int x = -1, y = -1;
  int res = glob_arr2[x][y]; // expected-warning{{garbage or undefined}}
}

void glob_invalid_index4() {
  int x = 3, y = 2;
  int res = glob_arr2[x][y]; // expected-warning{{garbage or undefined}}
}

const int glob_arr_no_init[10];
void glob_arr_index4() {
  // FIXME: Should warn {{FALSE}}, since the array has a static storage.
  clang_analyzer_eval(glob_arr_no_init[2]); // expected-warning{{UNKNOWN}}
}

const int glob_arr3[];              // IncompleteArrayType
const int glob_arr3[4] = {1, 2, 3}; // ConstantArrayType
void glob_arr_index5() {
  clang_analyzer_eval(glob_arr3[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr3[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr3[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr3[3] == 0); // expected-warning{{TRUE}}
}

void glob_invalid_index5() {
  int x = 42;
  int res = glob_arr3[x]; // expected-warning{{garbage or undefined}}
}

void glob_invalid_index6() {
  int x = -42;
  int res = glob_arr3[x]; // expected-warning{{garbage or undefined}}
}

const int glob_arr4[];              // IncompleteArrayType
const int glob_arr4[4] = {1, 2, 3}; // ConstantArrayType
const int glob_arr4[];              // ConstantArrayType (according to AST)
void glob_arr_index6() {
  clang_analyzer_eval(glob_arr4[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[1] == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[2] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(glob_arr4[3] == 0); // expected-warning{{TRUE}}
}

void glob_invalid_index7() {
  int x = 42;
  int res = glob_arr4[x]; // expected-warning{{garbage or undefined}}
}

void glob_invalid_index8() {
  int x = -42;
  int res = glob_arr4[x]; // expected-warning{{garbage or undefined}}
}
