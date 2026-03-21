// RUN: %clang_cc1 -fsyntax-only -std=c++23 -Wlifetime-safety -Wno-dangling -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++23 -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety -Wno-dangling -verify %s

// expected-no-diagnostics
struct S {
  static S operator()(int, int&&);
};

void indexing_with_static_operator() {
  S()(1, 2);
}
