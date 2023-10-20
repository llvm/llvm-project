// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

// Make sure we correctly treat __typeof as potentially-evaluated when appropriate
template<typename T> void f(T n) {
  int buffer[n];
  [&buffer] { __typeof(buffer) x; }();
}
int main() {
  f<int>(1);
}
