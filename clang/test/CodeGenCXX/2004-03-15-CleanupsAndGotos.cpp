// RUN: %clang_cc1 -emit-llvm %s -o -

// Testcase from Bug 291

struct X {
  ~X();
};

void foo() {
  X v;

TryAgain:
  goto TryAgain;
}
