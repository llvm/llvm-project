// RUN: %clang_analyze_cc1 -analyzer-checker=core -std=c++23 -verify %s
// expected-no-diagnostics

struct S {
  bool operator==(this auto, S) {
    return true;
  }
};
int use_deducing_this() {
  return S{} == S{};
}
