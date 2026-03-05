// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

unsigned long s(int i, unsigned long x) {
  return x << i;
}

// CHECK: cir.shift(left, %3 : !u64i, %4 : !s32i) -> !u64i