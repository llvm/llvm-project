// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s
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
  // CHECK:    [[RETVAL:%.*]] = alloca [[STRUCT_S:%.*]], align 4, addrspace(5)
  // CHECK-NEXT:    [[RETVAL_ASCAST:%.*]] = addrspacecast ptr addrspace(5) [[RETVAL]] to ptr
  S s;
  // CHECK: call void @{{.+}}bar{{.+}}(ptr {{.*}}[[S_REF:%.+]])
  bar(s);
  // CHECK:    call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[RETVAL_ASCAST]], ptr align 4 [[S]], i64 4, i1 false)
  // CHECK:    [[TMP0:%.*]] = load [[STRUCT_S]], ptr [[RETVAL_ASCAST]], align 4
  // CHECK:    ret [[STRUCT_S]] [[TMP0]]
  return s;
}
#pragma omp end declare target

#endif
