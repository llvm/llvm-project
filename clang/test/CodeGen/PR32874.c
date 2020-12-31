// RUN: %clang_cc1 -x c -S -emit-llvm -o - -triple x86_64-apple-darwin10 %s \
// RUN:   -w -fsanitize=signed-integer-overflow,unsigned-integer-overflow,integer-divide-by-zero,float-divide-by-zero \
// RUN:   | FileCheck %s

// CHECK-LABEL: define{{.*}} void @foo
// CHECK-NOT: !nosanitize
void foo(const int *p) {
  // __builtin_prefetch expects its optional arguments to be constant integers.
  // Check that ubsan does not instrument any safe arithmetic performed in
  // operands to __builtin_prefetch. (A clang frontend check should reject
  // unsafe arithmetic in these operands.)

  __builtin_prefetch(p, 0 + 1, 0 + 3);
  __builtin_prefetch(p, 1 - 0, 3 - 0);
  __builtin_prefetch(p, 1 * 1, 1 * 3);
  __builtin_prefetch(p, 1 / 1, 3 / 1);
  __builtin_prefetch(p, 3 % 2, 3 % 1);

  __builtin_prefetch(p, 0U + 1U, 0U + 3U);
  __builtin_prefetch(p, 1U - 0U, 3U - 0U);
  __builtin_prefetch(p, 1U * 1U, 1U * 3U);
  __builtin_prefetch(p, 1U / 1U, 3U / 1U);
  __builtin_prefetch(p, 3U % 2U, 3U % 1U);
}

// CHECK-LABEL: define{{.*}} void @ub_constant_arithmetic
void ub_constant_arithmetic() {
  // Check that we still instrument unsafe arithmetic, even if it is known to
  // be unsafe at compile time.

  int INT_MIN = 0xffffffff;
  int INT_MAX = 0x7fffffff;

  // CHECK: call void @__ubsan_handle_add_overflow
  // CHECK: call void @__ubsan_handle_add_overflow
  INT_MAX + 1;
  INT_MAX + -1;

  // CHECK: call void @__ubsan_handle_negate_overflow
  // CHECK: call void @__ubsan_handle_sub_overflow
  -INT_MIN;
  -INT_MAX - 2;

  // CHECK: call void @__ubsan_handle_mul_overflow
  // CHECK: call void @__ubsan_handle_mul_overflow
  INT_MAX * INT_MAX;
  INT_MIN * INT_MIN;

  // CHECK: call void @__ubsan_handle_divrem_overflow
  // CHECK: call void @__ubsan_handle_divrem_overflow
  1 / 0;
  INT_MIN / -1;

  // CHECK: call void @__ubsan_handle_divrem_overflow
  // CHECK: call void @__ubsan_handle_divrem_overflow
  1 % 0;
  INT_MIN % -1;

  // CHECK: call void @__ubsan_handle_divrem_overflow
  1.0 / 0.0;
}
