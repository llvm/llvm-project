// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 2 target regions on the gpu is set to non-SPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l21}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l26}}_exec_mode = weak constant i8 1

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target parallel map(tofrom: aa) num_threads(1024)
  {
    aa += 1;
  }

  #pragma omp target parallel map(tofrom:a, aa, b) if(target: n>40) num_threads(n)
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

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l21}}(
  // CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
  // CHECK: store i16* {{%.+}}, i16** [[AA_ADDR]], align
  // CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
  // CHECK: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK: call void @__kmpc_kernel_init(i32
  // CHECK: call void @__kmpc_push_num_threads
  // CHECK: call void @__kmpc_kernel_deinit(i16 1)
  // CHECK: ret void
  // CHECK: }

  // CHECK: define internal void @{{.+}}(i32* noalias %{{.+}}, i32* noalias %{{.+}}, i16* {{[^%]*}}[[ARG:%.+]])
  // CHECK: = alloca i32*, align
  // CHECK: = alloca i32*, align
  // CHECK: [[AA_ADDR:%.+]] = alloca i16*, align
  // CHECK: store i16* [[ARG]], i16** [[AA_ADDR]], align
  // CHECK: [[AA:%.+]] = load i16*, i16** [[AA_ADDR]], align
  // CHECK: [[VAL:%.+]] = load i16, i16* [[AA]], align
  // CHECK: store i16 {{%.+}}, i16* [[AA]], align
  // CHECK: ret void
  // CHECK: }






  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l26}}(
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
  // CHECK: call void @__kmpc_kernel_init(i32
  // CHECK: call void @__kmpc_push_num_threads
  // CHECK: call void @__kmpc_kernel_deinit(i16 1)
  // CHECK: ret void
  // CHECK: }

  // CHECK: define internal void @{{.+}}(i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* {{[^%]*}}[[ARG1:%.+]], i16* {{[^%]*}}[[ARG2:%.+]], [10 x i32]* {{[^%]*}}[[ARG3:%.+]])
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
