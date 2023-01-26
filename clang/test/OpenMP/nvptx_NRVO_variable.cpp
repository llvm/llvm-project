// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

struct S {
  int a;
  S() : a(1) {}
};

#pragma omp declare target
void bar(S &);
// CHECK-LABEL: foo
S foo() {
  // CHECK: [[RETVAL:%.+]] = alloca %struct.S,
  S s;
  // CHECK: call void @{{.+}}bar{{.+}}(ptr {{.*}}[[S_REF:%.+]])
  bar(s);
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}[[RETVAL]], ptr {{.*}}[[S_REF]], i64 4, i1 false)
  // CHECK: [[VAL:%.+]] = load %struct.S, ptr [[RETVAL]],
  // CHECK: ret %struct.S [[VAL]]
  return s;
}
#pragma omp end declare target

#endif
