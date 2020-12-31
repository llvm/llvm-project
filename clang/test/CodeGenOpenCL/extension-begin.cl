// RUN: %clang_cc1 %s -triple spir-unknown-unknown -emit-llvm -o - | FileCheck %s

__attribute__((overloadable)) void f(int x);

#pragma OPENCL EXTENSION my_ext : begin

__attribute__((overloadable)) void f(long x);

#pragma OPENCL EXTENSION my_ext : end

#pragma OPENCL EXTENSION my_ext : enable

//CHECK: define{{.*}} spir_func void @test_f1(i64 %x)
//CHECK: call spir_func void @_Z1fl(i64 %{{.*}})
void test_f1(long x) {
  f(x);
}

#pragma OPENCL EXTENSION my_ext : disable

//CHECK: define{{.*}} spir_func void @test_f2(i64 %x)
//CHECK: call spir_func void @_Z1fi(i32 %{{.*}})
void test_f2(long x) {
  f(x);
}
