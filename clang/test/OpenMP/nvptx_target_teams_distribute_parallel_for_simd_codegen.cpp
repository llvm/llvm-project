// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 4 target regions on the gpu is set to SPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l30}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l36}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l41}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l46}}_exec_mode = weak constant i8 0

#define N 1000
#define M 10

template<typename tx>
tx ftemplate(int n) {
  tx a[N];
  short aa[N];
  tx b[10];
  tx c[M][M];
  tx f = n;
  tx l;
  int k;

#pragma omp target teams distribute parallel for simd lastprivate(l) dist_schedule(static,128) schedule(static,32)
  for(int i = 0; i < n; i++) {
    a[i] = 1;
    l = i;
  }

 #pragma omp target teams distribute parallel for simd map(tofrom: aa) num_teams(M) thread_limit(64)
  for(int i = 0; i < n; i++) {
    aa[i] += 1;
  }

#pragma omp target teams distribute parallel for simd map(tofrom:a, aa, b) if(target: n>40) proc_bind(spread)
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

#pragma omp target teams distribute parallel for simd collapse(2) firstprivate(f) private(k)
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < M; j++) {
      k = M;
      c[i][j] = i+j*f+k;
    }
  }

  return a[0];
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// CHECK-DAG: [[MEM_TY:%.+]] = type { [128 x i8] }
// CHECK-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// CHECK-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// CHECK-DAG: [[KERNEL_SIZE:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// CHECK-DAG: [[KERNEL_SHARED:@.+]] = internal unnamed_addr constant i16 1

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}_l30(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_get_team_static_memory(i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} 4, i16 1, i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// CHECK: [[TEAM_ALLOC:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// CHECK: [[BC:%.+]] = bitcast i8* [[TEAM_ALLOC]] to [[REC:%.+]]*
// CHECK: getelementptr inbounds [[REC]], [[REC]]* [[BC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL1:@.+]](
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_restore_team_static_memory(i16 1)
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: define internal void [[OUTL1]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL2:@.+]](
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: define internal void [[OUTL2]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL3:@.+]](
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: define internal void [[OUTL3]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define {{.*}}void {{@__omp_offloading_.+}}({{.+}}, i{{32|64}} [[F_IN:%.+]])
// CHECK: store {{.+}} [[F_IN]], {{.+}}* {{.+}},
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: store {{.+}} 99, {{.+}}* [[COMB_UB:%.+]], align
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91, {{.+}}, {{.+}}, {{.+}}* [[COMB_UB]],
// CHECK: {{call|invoke}} void [[OUTL4:@.+]](
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: define internal void [[OUTL4]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

#endif
