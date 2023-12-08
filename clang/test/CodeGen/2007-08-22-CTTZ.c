// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int bork(unsigned long long x) {
  // CHECK: llvm.cttz.i64
  // CHECK: llvm.cttz.i64
  // CHECK-NOT: lshr
  return __builtin_ctzll(x);
}
