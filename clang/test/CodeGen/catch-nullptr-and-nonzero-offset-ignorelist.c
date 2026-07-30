// RUN: %clang_cc1 -x c -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow"
// RUN: %clang_cc1 -x c -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow"

// RUN: %clang_cc1 -x c++ -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow"
// RUN: %clang_cc1 -x c++ -fno-delete-null-pointer-checks -fsanitize=pointer-overflow -fsanitize-recover=pointer-overflow -emit-llvm %s -o - -triple x86_64-linux-gnu | FileCheck %s -implicit-check-not="call void @__ubsan_handle_pointer_overflow"

#ifdef __cplusplus
extern "C" {
#endif

// CHECK-LABEL: @baseline
char *baseline(char *base, unsigned long offset) {
  // CHECK: call void @__ubsan_handle_pointer_overflow(
  return base + offset;
}

// CHECK-LABEL: @ignorelist_0
__attribute__((no_sanitize("undefined"))) char *ignorelist_0(char *base, unsigned long offset) {
  return base + offset;
}

// CHECK-LABEL: @ignorelist_1
__attribute__((no_sanitize("pointer-overflow"))) char *ignorelist_1(char *base, unsigned long offset) {
  return base + offset;
}

// CHECK-LABEL: @ignore_non_default_address_space
__attribute__((address_space(1))) char *ignore_non_default_address_space(__attribute__((address_space(1))) char *base, unsigned long offset) {
  return base + offset;
}

#ifdef __cplusplus
}
#endif
