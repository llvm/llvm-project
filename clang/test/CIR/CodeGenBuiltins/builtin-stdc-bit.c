// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

void test_stdc_trailing_zeros(unsigned long long x) {
  int cnt = __builtin_stdc_trailing_zeros(x);
  (void)cnt;
}

// CHECK-LABEL: test_stdc_trailing_zeros
// CHECK: cir.ctz
// CHECK-NOT: poison_zero
