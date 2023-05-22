// RUN: %clang_cc1 -emit-llvm -triple x86_64 -std=c17 -fsanitize=function %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} @call_no_prototype(
// CHECK-NOT:     __ubsan_handle_function_type_mismatch
void call_no_prototype(void (*f)()) { f(); }

// CHECK-LABEL: define{{.*}} @call_prototype(
// CHECK:         __ubsan_handle_function_type_mismatch
void call_prototype(void (*f)(void)) { f(); }
