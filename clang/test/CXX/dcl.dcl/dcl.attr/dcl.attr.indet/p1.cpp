// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -pedantic -verify=cxx23 %s

#if __cplusplus <= 202302L
void cxx23() {
  [[indeterminate]] int x; // cxx23-warning {{use of the 'indeterminate' attribute is a C++26 extension}}
}
#else
struct S {
  int m;
  S() {}
};

void block_variables() {
  [[indeterminate]] int a;
  [[indeterminate]] int b[4], c;
  [[indeterminate]] S s;
  int d [[indeterminate]];
  for ([[indeterminate]] int i;;)
    break;

  [[indeterminate]] static int e;       // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] thread_local int f; // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate]] extern int g;       // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
  [[indeterminate(1)]] int h;           // expected-error {{'indeterminate' attribute takes no arguments}}
}

void parameter([[indeterminate]] int p);
auto lambda = []([[indeterminate]] int p) {};

struct M {
  void member_function([[indeterminate]] int p);
  [[indeterminate]] int m; // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
};

[[indeterminate]] int global;  // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}
[[indeterminate]] void func(); // expected-warning {{'indeterminate' attribute only applies to local variables or function parameters}}

template <typename T> void dependent([[indeterminate]] T p) {
  [[indeterminate]] T v;
}
template void dependent<int>(int);
template void dependent<S>(S);
#endif
