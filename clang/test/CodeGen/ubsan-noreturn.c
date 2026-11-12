// RUN: %clang_cc1 %s -emit-llvm -fsanitize=unreachable -o - | FileCheck %s

// CHECK-LABEL: @f(
void __attribute__((noreturn)) f(void) {
  // CHECK: __ubsan_handle_builtin_unreachable
  // CHECK: unreachable
}
