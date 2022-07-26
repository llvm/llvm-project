// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

void clang_analyzer_eval(bool);

void array_value_a(void) {
  int arr[2];
  auto [a, b] = arr;
  arr[0] = 0;

  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_value_b(void) {
  int arr[] = {1, 2};
  auto [a, b] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}

  int x = a; // no-warning
}

void array_value_c(void) {
  int arr[3];

  arr[1] = 1;

  auto [a, b, c] = arr;

  clang_analyzer_eval(b == arr[1]); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_value_d(void) {
  int arr[3];

  arr[1] = 1;

  auto [a, b, c] = arr;

  clang_analyzer_eval(b == arr[1]); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = c; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_value_e(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto [i, j] = init;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // no-warning
}

void array_value_f(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto [i, j] = uninit;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_lref_a(void) {
  int arr[2];
  auto &[a, b] = arr;
  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_lref_b(void) {
  int arr[] = {1, 2};
  auto &[a, b] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}

  int x = a; // no-warning
}

void array_lref_c(void) {
  int arr[2];
  auto &[a, b] = arr;

  arr[0] = 1;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}

  int x = a; // no-warning
  int y = b; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_lref_d(void) {
  int arr[3];

  arr[1] = 1;

  auto &[a, b, c] = arr;

  clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_lref_e(void) {
  int arr[3];

  arr[1] = 1;

  auto &[a, b, c] = arr;

  clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = c; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_lref_f(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto &[i, j] = init;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // no-warning
}

void array_lref_g(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto &[i, j] = uninit;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_rref_a(void) {
  int arr[2];
  auto &&[a, b] = arr;
  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_rref_b(void) {
  int arr[] = {1, 2};
  auto &&[a, b] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}

  int x = a; // no-warning
}

void array_rref_c(void) {
  int arr[2];
  auto &&[a, b] = arr;

  arr[0] = 1;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}

  int x = a; // no-warning
  int y = b; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_rref_d(void) {
  int arr[3];

  arr[1] = 1;

  auto &&[a, b, c] = arr;

  clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_rref_e(void) {
  int arr[3];

  arr[1] = 1;

  auto &&[a, b, c] = arr;

  clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}

  int y = b; // no-warning
  int x = c; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_rref_f(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto &&[i, j] = init;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}
  clang_analyzer_eval(j == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // no-warning
}

void array_rref_g(void) {
  int uninit[2];
  int init[2] = {0};

  uninit[0] = init[0];

  auto &&[i, j] = uninit;

  clang_analyzer_eval(i == 0); // expected-warning{{TRUE}}

  int a = i; // no-warning
  int b = j; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_change_a(void) {
  int arr[] = {1, 2};

  auto [a, b] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  a = 3;
  clang_analyzer_eval(a == 3); // expected-warning{{TRUE}}

  clang_analyzer_eval(arr[0] == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(arr[1] == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
}

void array_change_b(void) {
  int arr[] = {1, 2};

  auto &[a, b] = arr;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}

  a = 3;
  clang_analyzer_eval(a == 3); // expected-warning{{TRUE}}

  clang_analyzer_eval(arr[0] == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(arr[1] == 2); // expected-warning{{TRUE}}
}

void array_small_a(void) {
  int arr[5];

  auto [a, b, c, d, e] = arr;

  int x = e; // expected-warning{{Assigned value is garbage or undefined}}
}

void array_big_a(void) {
  int arr[6];

  auto [a, b, c, d, e, f] = arr;

  // FIXME: These will be Undefined when we handle reading Undefined values from lazyCompoundVal.
  clang_analyzer_eval(a == 1); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(b == 2); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(c == 3); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(d == 4); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(e == 5); // expected-warning{{UNKNOWN}}
  clang_analyzer_eval(f == 6); // expected-warning{{UNKNOWN}}
}

struct S {
  int a = 1;
  int b = 2;
};

void non_pod_val(void) {
  S arr[2];

  auto [x, y] = arr;

  clang_analyzer_eval(x.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 2); // expected-warning{{TRUE}}
}

void non_pod_val_syntax_2(void) {
  S arr[2];

  auto [x, y](arr);

  clang_analyzer_eval(x.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 2); // expected-warning{{TRUE}}
}

void non_pod_lref(void) {
  S arr[2];

  auto &[x, y] = arr;

  clang_analyzer_eval(x.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 2); // expected-warning{{TRUE}}
}

void non_pod_rref(void) {
  S arr[2];

  auto &&[x, y] = arr;

  clang_analyzer_eval(x.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 2); // expected-warning{{TRUE}}
}

struct SUD {
  inline static int c = 0;

  int a = 1;
  int b = 2;

  SUD() { ++c; };

  SUD(const SUD &copy) {
    a = copy.a + 1;
    b = copy.b + 1;
  }
};

void non_pod_user_defined_val(void) {
  SUD arr[2];

  auto [x, y] = arr;

  clang_analyzer_eval(x.a == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 3); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 3); // expected-warning{{TRUE}}
}

void non_pod_user_defined_val_syntax_2(void) {
  SUD::c = 0;
  SUD arr[2];

  auto [x, y](arr);

  clang_analyzer_eval(SUD::c == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(x.a == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(x.b == 3); // expected-warning{{TRUE}}

  clang_analyzer_eval(y.a == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(y.b == 3); // expected-warning{{TRUE}}
}
