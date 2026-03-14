// RUN: %clang_cc1 -emit-llvm-only %s

namespace PR43080 {
  int f(int i) { return sizeof i<i; }
}

namespace PR42861 {
  const unsigned long s = alignof(int);
  void foo() {  alignas(s) int j; }
}
