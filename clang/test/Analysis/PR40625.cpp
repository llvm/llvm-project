// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=true

void f(const int *end);

void g(const int (&arrr)[10]) {
  f(arrr); // expected-warning{{1st function call argument is a pointer to uninitialized value}}
}

void h() {
  int arr[10];

  g(arr);
}
