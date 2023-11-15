// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify -fsyntax-only %s
// REQUIRES: x86-registered-target
// expected-no-diagnostics

// Testcase for https://github.com/llvm/llvm-project/issues/69717

#pragma float_control(precise, on, push)

template<typename T>
constexpr T multi(T x, T y) {
  return x * y;
}

int multi_i(int x, int y) {
  return multi<int>(x, y);
}

#pragma float_control(pop)
