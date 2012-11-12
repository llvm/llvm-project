// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Make sure we correctly treat __typeof as potentially-evaluated when appropriate
template<typename T> void f(T n) {
  int buffer[n]; // expected-note {{declared here}}
  [] { __typeof(buffer) x; }(); // expected-error {{variable 'buffer' with variably modified type cannot be captured in a lambda expression}}
}
int main() {
  f<int>(1); // expected-note {{in instantiation}}
}
