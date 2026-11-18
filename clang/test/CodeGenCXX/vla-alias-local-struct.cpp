// RUN: %clang_cc1 -std=gnu++20 -emit-llvm %s -o - >/dev/null

int f() { return 1; }

int test0() {
  using X = int[f()];

  struct S {
    S() {
      X x;
    }
  } s;

  return 0;
}
