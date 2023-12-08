// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// Make sure we correctly treat __typeof as potentially-evaluated when appropriate
template<typename T> void f(T n) { // expected-note {{declared here}}
  int buffer[n]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                    expected-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  [&buffer] { __typeof(buffer) x; }();
}
int main() {
  f<int>(1); // expected-note {{in instantiation of function template specialization 'f<int>' requested here}}
}
