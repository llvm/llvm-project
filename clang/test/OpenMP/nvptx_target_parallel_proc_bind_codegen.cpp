// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 3 target regions on the gpu is set to SPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l29}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l33}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l38}}_exec_mode = weak constant i8 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target parallel proc_bind(master)
  {
  }

  #pragma omp target parallel proc_bind(spread)
  {
    aa += 1;
  }

  #pragma omp target parallel proc_bind(close)
  {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l29}}(
// CHECK: call void @__kmpc_spmd_kernel_init(i32 {{.+}}, i16 1)
// CHECK: br label {{%?}}[[EXEC:.+]]
//
// CHECK: [[EXEC]]
// CHECK-NOT: call void @__kmpc_push_proc_bind
// CHECK: {{call|invoke}} void [[OP1:@.+]](
// CHECK: br label {{%?}}[[DONE:.+]]
//
// CHECK: [[DONE]]
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: br label {{%?}}[[EXIT:.+]]
//
// CHECK: [[EXIT]]
// CHECK: ret void
// CHECK: }

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l33}}(
// CHECK: call void @__kmpc_spmd_kernel_init(i32 {{.+}}, i16 1)
// CHECK: br label {{%?}}[[EXEC:.+]]
//
// CHECK: [[EXEC]]
// CHECK-NOT: call void @__kmpc_push_proc_bind
// CHECK: {{call|invoke}} void [[OP1:@.+]](
// CHECK: br label {{%?}}[[DONE:.+]]
//
// CHECK: [[DONE]]
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: br label {{%?}}[[EXIT:.+]]
//
// CHECK: [[EXIT]]
// CHECK: ret void
// CHECK: }

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l38}}(
// CHECK: call void @__kmpc_spmd_kernel_init(i32 {{.+}}, i16 1)
// CHECK: br label {{%?}}[[EXEC:.+]]
//
// CHECK: [[EXEC]]
// CHECK-NOT: call void @__kmpc_push_proc_bind
// CHECK: {{call|invoke}} void [[OP1:@.+]](
// CHECK: br label {{%?}}[[DONE:.+]]
//
// CHECK: [[DONE]]
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: br label {{%?}}[[EXIT:.+]]
//
// CHECK: [[EXIT]]
// CHECK: ret void
// CHECK: }
#endif
