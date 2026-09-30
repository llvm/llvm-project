// RUN: %clang_cc1 -std=c++20 -emit-llvm -o %t.ll -Wno-stack-exhausted -fconstexpr-depth=1024 %s
// RUN: %clang_cc1 -std=c++20 -emit-llvm -o %t.new.ll -Wno-stack-exhausted -fconstexpr-depth=1024 -fexperimental-new-constant-interpreter %s

// Regression test for GH200673. IR generation attempts to evaluate the call
// and must not exhaust the stack before reaching the constexpr call limit.
constexpr double a(double b) {
  return 0 - b * -a(b) * 2 * 2 * 0 * 0 * 2 * 2 * 0 * 2 / 0 * 0 * 1 / 0 * 2 * 0 /
                 0 * 0 * 1 / 0 * 0 / 0 / b * b * b * 0 / 0 * 0 * 1 / 0 * 0 * 0 *
                 2 * 0 / b;
}

int c() {
  double b;
  b ? a(0) : 0;
  return 0;
}
