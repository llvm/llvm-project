// RUN: %clang_cc1 -std=c++11 -verify %s

// https://github.com/llvm/llvm-project/issues/185693
// Explicitly deleted function template specializations were incorrectly
// reported as "implicitly deleted" in overload resolution diagnostics.

template <typename T> void fred(const T &x);
template <> void fred(const double &) = delete; // expected-note {{explicitly deleted}}

int main() {
  fred(8.0); // expected-error {{call to deleted function 'fred'}}
}
