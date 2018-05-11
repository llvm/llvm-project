// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 2 target regions on the gpu is set to NonSPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l25}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l30}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l35}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l40}}_exec_mode = weak constant i8 1

#define N 1000

template<typename tx>
tx ftemplate(int n) {
  tx a[N];
  short aa[N];
  tx b[10];

  #pragma omp target simd
  for(int i = 0; i < n; i++) {
    a[i] = 1;
  }

  #pragma omp target simd
  for (int i = 0; i < n; i++) {
    aa[i] += 1;
  }

  #pragma omp target simd
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

  #pragma omp target simd reduction(+:n)
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

  return a[0];
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+l25}}(
// CHECK: call void @__kmpc_kernel_init(i32 %{{.+}}, i16 1)
// CHECK-NOT: call void @__kmpc_for_static_init
// CHECK-NOT: call void @__kmpc_for_static_fini
// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+l30}}(
// CHECK: call void @__kmpc_kernel_init(i32 %{{.+}}, i16 1)
// CHECK-NOT: call void @__kmpc_for_static_init
// CHECK-NOT: call void @__kmpc_for_static_fini
// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+l35}}(
// CHECK: call void @__kmpc_kernel_init(i32 %{{.+}}, i16 1)
// CHECK-NOT: call void @__kmpc_for_static_init
// CHECK-NOT: call void @__kmpc_for_static_fini
// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+l40}}(
// CHECK: call void @__kmpc_kernel_init(i32 %{{.+}}, i16 1)
// CHECK-NOT: call void @__kmpc_for_static_init
// CHECK-NOT: call void @__kmpc_for_static_fini
// CHECK: [[RES:%.+]] = call i32 @__kmpc_nvptx_simd_reduce_nowait(i32 %{{.+}}, i32 1, i{{64|32}} {{8|4}}, i8* %{{.+}}, void (i8*, i16, i16, i16)* @{{.+}}, void (i8*, i32)* @{{.+}})
// CHECK: switch i32 [[RES]]
// CHECK: call void @__kmpc_nvptx_end_reduce_nowait(i32 %{{.+}})
// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: ret void


#endif
