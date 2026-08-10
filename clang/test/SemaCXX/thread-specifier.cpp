// RUN: %clang_cc1 -fsyntax-only -std=c++23 -verify %s

namespace GH140509 {
template <typename T>
void not_instantiated() {
  static __thread T my_wrapper;
}

template <typename T>
void instantiated() {
  static __thread T my_wrapper = T{}; // expected-error {{initializer for thread-local variable must be a constant expression}} \
                                         expected-note {{use 'thread_local' to allow this}}
}

template <typename T>
void nondependent_var() {
  // Verify that the dependence of the initializer is what really matters.
  static __thread int my_wrapper = T{};
}

struct S {
  S() {}
};

void f() {
  instantiated<int>();
  instantiated<S>(); // expected-note {{in instantiation of function template specialization 'GH140509::instantiated<GH140509::S>' requested here}}
  nondependent_var<int>();
}
} // namespace GH140509
