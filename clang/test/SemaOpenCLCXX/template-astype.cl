// RUN: %clang_cc1 -fdeclare-opencl-builtins -finclude-default-header %s -cl-std=clc++ -verify

// Test as_type, which is defined in terms of __builtin_astype.
template <typename T>
auto templated_astype(T x) {
  return as_int2(x);
  // expected-error@-1{{invalid reinterpretation: sizes of 'int2' (vector of 2 'int' values) and '__private int' must match}}
}

auto test_long(long x) { return templated_astype(x); }

auto neg_test_int(int x) { return templated_astype(x); }
// expected-note@-1{{in instantiation of function template specialization 'templated_astype<int>' requested here}}

auto test_short4(short4 x) { return templated_astype(x); }

// Test __builtin_astype.
template <typename T>
auto templated_builtin_astype(T x) {
  return __builtin_astype(x, int2);
}

auto test_builtin(char8 x) { return templated_builtin_astype(x); }
