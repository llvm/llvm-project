// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -o - -triple spirv64 %s | FileCheck %s

// Check that we have a way to recover pointer
// types for extern function prototypes (see PR56660).
extern void foo(global int * ptr);
kernel void k(global int * ptr) {
  foo(ptr);
}
//CHECK: define spir_kernel void @k(i32 {{.*}}*
//CHECK: declare spir_func void @foo(i32 {{.*}}*
