// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsanitize=null,alignment,object-size,vptr -S -emit-llvm %s -o - | FileCheck %s

// CHECK: @volatile_null_deref
void volatile_null_deref(volatile int *p) {
  // CHECK-NOT: call{{.*}}ubsan
  *p;
}
