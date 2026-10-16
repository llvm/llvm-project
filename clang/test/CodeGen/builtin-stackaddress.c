// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define {{[^@]+}} @a()
// CHECK: call {{[^@]+}} @llvm.stackaddress.p0()
void *a() {
  return __builtin_stack_address();
}
