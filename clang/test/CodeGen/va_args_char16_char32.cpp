// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s

__builtin_va_list ap;

// CHECK-LABEL: define {{.*}} @_Z3foov
void foo() {
  __builtin_va_arg(ap, char16_t);
  // CHECK: %vaarg.addr = phi ptr
  // CHECK-NEXT: %{{.*}} = load i32, ptr %vaarg.addr

  __builtin_va_arg(ap, char32_t);
  // CHECK: %vaarg.addr{{.*}} = phi ptr
  // CHECK-NEXT: %{{.*}} = load i32, ptr %vaarg.addr
}
