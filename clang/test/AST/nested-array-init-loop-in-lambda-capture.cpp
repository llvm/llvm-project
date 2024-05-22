// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++17 -verify=ref %s

// ref-no-diagnostics
// expected-no-diagnostics

void used_to_crash() {
  int s[2][2];

  int arr[4];

  arr[0] = [s] { return s[0][0]; }();
}
