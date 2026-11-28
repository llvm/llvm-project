// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -verify -verify-ignore-unexpected=note %s
// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -verify %s
// RUN: %clang_cc1 -std=c++20 -verify=off %s
// off-no-diagnostics

int *begin() {
  static int values[3];
  return values;
}

// Only the definition owns the initializer.
extern int *second;
int *second = begin() + 1; // expected-warning {{unsafe pointer arithmetic}}
extern int *second;

// The enclosing function already analyzes local static initializers.
int *localSecond() {
  static int *p = begin() + 1; // expected-warning {{unsafe pointer arithmetic}}
  return p;
}

// The capture initializer and the lambda body each need one diagnostic.
auto getThird = [second = begin() + 1] { // expected-warning {{unsafe pointer arithmetic}}
  (void)second;
  return begin() + 2; // expected-warning {{unsafe pointer arithmetic}}
};
