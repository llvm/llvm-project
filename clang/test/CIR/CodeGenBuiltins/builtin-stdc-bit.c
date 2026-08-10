// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -I%S/../../CodeGen/Inputs -fclangir -emit-cir %s -o - | FileCheck %s

#include <stdbit.h>

void test_stdc_trailing_zeros(unsigned long long x) {
  int cnt = __builtin_stdc_trailing_zeros(x);
  (void)cnt;
}

// CHECK-LABEL: test_stdc_trailing_zeros
// CHECK: cir.ctz
// CHECK-NOT: poison_zero
// CHECK: cir.return

unsigned test_stdc_leading_zeros(unsigned x) {
  return __builtin_stdc_leading_zeros(x);
}

// CHECK-LABEL: test_stdc_leading_zeros
// CHECK: cir.clz
// CHECK-NOT: poison_zero
// CHECK: cir.return

unsigned test_stdc_leading_zeros_ui(unsigned x) {
  return stdc_leading_zeros_ui(x);
}

// CHECK-LABEL: test_stdc_leading_zeros_ui
// CHECK: cir.clz
// CHECK-NOT: poison_zero
// CHECK: cir.return

unsigned test_stdc_count_ones(unsigned x) {
  return __builtin_stdc_count_ones(x);
}

// CHECK-LABEL: test_stdc_count_ones
// CHECK: cir.popcount
// CHECK: cir.return

unsigned test_stdc_count_ones_ui(unsigned x) {
  return stdc_count_ones_ui(x);
}

// CHECK-LABEL: test_stdc_count_ones_ui
// CHECK: cir.popcount
// CHECK: cir.return

unsigned test_stdc_bit_width(unsigned x) {
  return __builtin_stdc_bit_width(x);
}

// CHECK-LABEL: test_stdc_bit_width
// CHECK: cir.clz
// CHECK-NOT: poison_zero
// CHECK: cir.sub
// CHECK: cir.return

unsigned test_stdc_bit_width_ui(unsigned x) {
  return stdc_bit_width_ui(x);
}

// CHECK-LABEL: test_stdc_bit_width_ui
// CHECK: cir.clz
// CHECK-NOT: poison_zero
// CHECK: cir.sub
// CHECK: cir.return
