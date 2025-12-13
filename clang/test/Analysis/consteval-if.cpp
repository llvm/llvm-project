// RUN: %clang_analyze_cc1 -std=c++23 -analyzer-checker=core -verify %s
// RUN: %clang_analyze_cc1 -std=c++26 -analyzer-checker=core -verify %s

void test_consteval() {
  if consteval {
    int *ptr = nullptr;
    *ptr = 42; // expected-warning{{Dereference of null pointer (loaded from variable 'ptr')}}
  }
}

void test_not_consteval() {
  if !consteval {
    int *ptr = nullptr;
    *ptr = 42; // expected-warning{{Dereference of null pointer (loaded from variable 'ptr')}}
  }
}
