// RUN(some-feature): %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s --check-prefix=GATED

int add(int a, int b) {
  return a + b;
}
