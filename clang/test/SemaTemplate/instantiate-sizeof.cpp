// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

// Make sure we handle contexts correctly with sizeof
template<typename T> void f(T n) {
  int buffer[n];
  [] { int x = sizeof(sizeof(buffer)); }();
}
int main() {
  f<int>(1);
}
