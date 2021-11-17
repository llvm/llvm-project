// RUN: %clang_cc1 -fsanitize=alignment -fsanitize-recover=alignment -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_alignment_assumption" --check-prefixes=CHECK

// CHECK-LABEL: @baseline
void *baseline(void *x) {
  // CHECK: call void @__ubsan_handle_alignment_assumption(
  return __builtin_assume_aligned(x, 1);
}

// CHECK-LABEL: ignorelist_0
__attribute__((no_sanitize("undefined"))) void *ignorelist_0(void *x) {
  return __builtin_assume_aligned(x, 1);
}

// CHECK-LABEL: ignorelist_1
__attribute__((no_sanitize("alignment"))) void *ignorelist_1(void *x) {
  return __builtin_assume_aligned(x, 1);
}

// CHECK-LABEL: dont_ignore_volatile_ptrs
void *dont_ignore_volatile_ptrs(void * volatile x) {
  // CHECK: call void @__ubsan_handle_alignment_assumption(
  return __builtin_assume_aligned(x, 1);
}

// CHECK-LABEL: ignore_volatiles
void *ignore_volatiles(volatile void * x) {
  return __builtin_assume_aligned(x, 1);
}
