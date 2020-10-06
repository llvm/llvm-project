// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 2 target regions on the gpu is set to SPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l33}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l38}}_exec_mode = weak constant i8 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target parallel if(target: 0)
  {
    a += 1;
  }

  #pragma omp target parallel map(tofrom: aa)
  {
    aa += 1;
  }

  #pragma omp target parallel map(tofrom:a, aa, b) if(target: n>40)
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

  // CHECK-NOT: define {{.*}}void {{@__omp_offloading_.+template.+l17}}

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l33}}(
// CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack
// CHECK: store i16* {{%.+}}, i16** [[AA_ADDR]], align
// CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 1)
// CHECK: call void @__kmpc_data_sharing_init_stack_spmd
// CHECK: br label {{%?}}[[EXEC:.+]]
//
// CHECK: [[EXEC]]
// CHECK: {{call|invoke}} void [[OP1:@.+]]({{.+}}, {{.+}}, i16* [[AA]])
// CHECK: br label {{%?}}[[DONE:.+]]
//
// CHECK: [[DONE]]
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: br label {{%?}}[[EXIT:.+]]
//
// CHECK: [[EXIT]]
// CHECK: ret void
// CHECK: }

// CHECK: define internal void [[OP1]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i16* {{[^%]*}}[[ARG:%.+]])
// CHECK: = alloca i32*, align
// CHECK: = alloca i32*, align
// CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK: store i16* [[ARG]], i16** [[AA_ADDR]], align
// CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK: [[VAL:%.+]] = load i16, i16* [[AA]], align
// CHECK: store i16 {{%.+}}, i16* [[AA]], align
// CHECK: ret void
// CHECK: }

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l38}}(
// CHECK: [[A_ADDR:%.+]] = alloca i32*, align
// CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK: [[B_ADDR:%.+]] = alloca [10 x i32]*, align
// CHECK: store i32* {{%.+}}, i32** [[A_ADDR]], align
// CHECK: store i16* {{%.+}}, i16** [[AA_ADDR]], align
// CHECK: store [10 x i32]* {{%.+}}, [10 x i32]** [[B_ADDR]], align
// CHECK: [[A:%.+]] = load i32*, i32** [[A_ADDR]], align
// CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK: [[B:%.+]] = load [10 x i32]*, [10 x i32]** [[B_ADDR]], align
// CHECK: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 1)
// CHECK: call void @__kmpc_data_sharing_init_stack_spmd
// CHECK: br label {{%?}}[[EXEC:.+]]
//
// CHECK: [[EXEC]]
// CHECK: {{call|invoke}} void [[OP2:@.+]]({{.+}}, {{.+}}, i32* [[A]], i16* [[AA]], [10 x i32]* [[B]])
// CHECK: br label {{%?}}[[DONE:.+]]
//
// CHECK: [[DONE]]
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: br label {{%?}}[[EXIT:.+]]
//
// CHECK: [[EXIT]]
// CHECK: ret void
// CHECK: }

// CHECK: define internal void [[OP2]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i32* {{[^%]*}}[[ARG1:%.+]], i16* {{[^%]*}}[[ARG2:%.+]], [10 x i32]* {{[^%]*}}[[ARG3:%.+]])
// CHECK: = alloca i32*, align
// CHECK: = alloca i32*, align
// CHECK: [[A_ADDR:%.+]] = alloca i32*, align
// CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
// CHECK: [[B_ADDR:%.+]] = alloca [10 x i32]*, align
// CHECK: store i32* [[ARG1]], i32** [[A_ADDR]], align
// CHECK: store i16* [[ARG2]], i16** [[AA_ADDR]], align
// CHECK: store [10 x i32]* [[ARG3]], [10 x i32]** [[B_ADDR]], align
// CHECK: [[A:%.+]] = load i32*, i32** [[A_ADDR]], align
// CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
// CHECK: [[B:%.+]] = load [10 x i32]*, [10 x i32]** [[B_ADDR]], align
// CHECK: store i32 {{%.+}}, i32* [[A]], align
// CHECK: store i16 {{%.+}}, i16* [[AA]], align
// CHECK: [[ELT:%.+]] = getelementptr inbounds [10 x i32], [10 x i32]* [[B]],
// CHECK: store i32 {{%.+}}, i32* [[ELT]], align
// CHECK: ret void
// CHECK: }
#endif
