// RUN: %clang_cc1 -no-opaque-pointers < %s -triple armv5e-none-linux-gnueabi -emit-llvm -O1 | FileCheck %s

// FIXME: This file should not be checking -O1 output.
// Ie, it is testing many IR optimizer passes as part of front-end verification.

enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
};

int *test_c11_atomic_fetch_add_int_ptr(_Atomic(int *) *p) {
  // CHECK: test_c11_atomic_fetch_add_int_ptr
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_add_4(i8* noundef {{%[0-9]+}}, i32 noundef 12, i32 noundef 5)
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int *test_c11_atomic_fetch_sub_int_ptr(_Atomic(int *) *p) {
  // CHECK: test_c11_atomic_fetch_sub_int_ptr
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_sub_4(i8* noundef {{%[0-9]+}}, i32 noundef 20, i32 noundef 5)
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int test_c11_atomic_fetch_add_int(_Atomic(int) *p) {
  // CHECK: test_c11_atomic_fetch_add_int
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_add_4(i8* noundef {{%[0-9]+}}, i32 noundef 3, i32 noundef 5)
  return __c11_atomic_fetch_add(p, 3, memory_order_seq_cst);
}

int test_c11_atomic_fetch_sub_int(_Atomic(int) *p) {
  // CHECK: test_c11_atomic_fetch_sub_int
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_sub_4(i8* noundef {{%[0-9]+}}, i32 noundef 5, i32 noundef 5)
  return __c11_atomic_fetch_sub(p, 5, memory_order_seq_cst);
}

int *fp2a(int **p) {
  // CHECK: @fp2a
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_sub_4(i8* noundef {{%[0-9]+}}, i32 noundef 4, i32 noundef 0)
  // Note, the GNU builtins do not multiply by sizeof(T)!
  return __atomic_fetch_sub(p, 4, memory_order_relaxed);
}

int test_atomic_fetch_add(int *p) {
  // CHECK: test_atomic_fetch_add
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_add_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_add(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_sub(int *p) {
  // CHECK: test_atomic_fetch_sub
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_sub_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_sub(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_and(int *p) {
  // CHECK: test_atomic_fetch_and
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_and_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_and(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_or(int *p) {
  // CHECK: test_atomic_fetch_or
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_or_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_or(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_xor(int *p) {
  // CHECK: test_atomic_fetch_xor
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_xor_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_xor(p, 55, memory_order_seq_cst);
}

int test_atomic_fetch_nand(int *p) {
  // CHECK: test_atomic_fetch_nand
  // CHECK: {{%[^ ]*}} = call i32 @__atomic_fetch_nand_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  return __atomic_fetch_nand(p, 55, memory_order_seq_cst);
}

int test_atomic_add_fetch(int *p) {
  // CHECK: test_atomic_add_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_add_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // CHECK: {{%[^ ]*}} = add i32 [[CALL]], 55
  return __atomic_add_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_sub_fetch(int *p) {
  // CHECK: test_atomic_sub_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_sub_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // CHECK: {{%[^ ]*}} = add i32 [[CALL]], -55
  return __atomic_sub_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_and_fetch(int *p) {
  // CHECK: test_atomic_and_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_and_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // CHECK: {{%[^ ]*}} = and i32 [[CALL]], 55
  return __atomic_and_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_or_fetch(int *p) {
  // CHECK: test_atomic_or_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_or_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // CHECK: {{%[^ ]*}} = or i32 [[CALL]], 55
  return __atomic_or_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_xor_fetch(int *p) {
  // CHECK: test_atomic_xor_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_xor_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // CHECK: {{%[^ ]*}} = xor i32 [[CALL]], 55
  return __atomic_xor_fetch(p, 55, memory_order_seq_cst);
}

int test_atomic_nand_fetch(int *p) {
  // CHECK: test_atomic_nand_fetch
  // CHECK: [[CALL:%[^ ]*]] = call i32 @__atomic_fetch_nand_4(i8* noundef {{%[0-9]+}}, i32 noundef 55, i32 noundef 5)
  // FIXME: We should not be checking optimized IR. It changes independently of clang.
  // FIXME-CHECK: [[AND:%[^ ]*]] = and i32 [[CALL]], 55
  // FIXME-CHECK: {{%[^ ]*}} = xor i32 [[AND]], -1
  return __atomic_nand_fetch(p, 55, memory_order_seq_cst);
}
