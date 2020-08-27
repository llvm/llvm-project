// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix CHECK-DIV64 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix CHECK-DIV64 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -fopenmp-optimistic-collapse -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-DIV32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -fopenmp-optimistic-collapse -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-DIV32 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix CHECK-DIV64 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix CHECK-DIV64 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -fopenmp-optimistic-collapse -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-DIV32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -fopenmp-optimistic-collapse -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-DIV32 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 5 target regions on the gpu is set to SPMD Mode.
// CHECK-DAG: {{@__omp_offloading_.+l50}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l56}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l61}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l66}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l74}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l81}}_exec_mode = weak constant i8 0

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
  tx *v;

#pragma omp target teams distribute parallel for lastprivate(l) dist_schedule(static,128) schedule(static,32)
  for(int i = 0; i < n; i++) {
    a[i] = 1;
    l = i;
  }

#pragma omp target teams distribute parallel for map(tofrom: aa) num_teams(M) thread_limit(64)
  for(int i = 0; i < n; i++) {
    aa[i] += 1;
  }

#pragma omp target teams distribute parallel for map(tofrom:a, aa, b) if(target: n>40) proc_bind(spread)
  for(int i = 0; i < 10; i++) {
    b[i] += 1;
  }

#pragma omp target teams distribute parallel for collapse(2) firstprivate(f) private(k)
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < M; j++) {
      k = M;
      c[i][j] = i + j * f + k;
    }
  }

#pragma omp target teams distribute parallel for collapse(2)
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      c[i][j] = i + j;
    }
  }

#pragma omp target teams distribute parallel for map(a, v[:N])
  for(int i = 0; i < n; i++)
    a[i] = v[i];
  return a[0];
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// SEQ-DAG: [[MEM_TY:%.+]] = type { [128 x i8] }
// SEQ-DAG: [[SHARED_GLOBAL_RD:@.+]] = common addrspace(3) global [[MEM_TY]] zeroinitializer
// SEQ-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* null
// SEQ-DAG: [[KERNEL_SIZE:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// SEQ-DAG: [[KERNEL_SHARED:@.+]] = internal unnamed_addr constant i16 1

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}_l50(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void [[PARALLEL:@.+]](
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)

// CHECK: define internal void [[PARALLEL]](
// SEQ: [[SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// SEQ: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE]],
// SEQ: call void @__kmpc_get_team_static_memory(i16 1, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// SEQ: [[TEAM_ALLOC:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// SEQ: [[ADDR:%.+]] = getelementptr inbounds i8, i8* [[TEAM_ALLOC]], i{{64|32}} 0
// PAR: [[ADDR:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i{{32|64}} 4, i16 1)
// CHECK: [[BC:%.+]] = bitcast i8* [[ADDR]] to [[REC:%.+]]*
// CHECK: getelementptr inbounds [[REC]], [[REC]]* [[BC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL1:@.+]](
// CHECK: call void @__kmpc_for_static_fini(
// SEQ: [[SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// SEQ: call void @__kmpc_restore_team_static_memory(i16 1, i16 [[SHARED]])
// PAR: call void @__kmpc_data_sharing_pop_stack(i8* [[ADDR]])
// CHECK: ret void

// CHECK: define internal void [[OUTL1]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL2:@.+]](
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: define internal void [[OUTL2]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+}}(
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91,
// CHECK: {{call|invoke}} void [[OUTL3:@.+]](
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: define internal void [[OUTL3]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// Distribute with collapse(2)
// CHECK: define {{.*}}void {{@__omp_offloading_.+}}({{.+}}, i{{32|64}} [[F_IN:%.+]])
// CHECK-DAG: [[THREAD_LIMIT:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK: call void @__kmpc_spmd_kernel_init(i32 [[THREAD_LIMIT]], i16 0, i16 0)
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 0)
// CHECK: ret void

// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: [[OMP_IV:%.+]] = alloca
// CHECK: alloca
// CHECK: alloca
// CHECK: [[OMP_LB:%.+]] = alloca
// CHECK: [[OMP_UB:%.+]] = alloca
// CHECK: [[OMP_ST:%.+]] = alloca
// CHECK: store {{.+}} [[F_IN]], {{.+}}* {{.+}},
// CHECK: store {{.+}} 99, {{.+}}* [[COMB_UB:%.+]], align
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 91, {{.+}}, {{.+}}, {{.+}}* [[COMB_UB]],

// check EUB for distribute
// CHECK-DAG: [[OMP_UB_VAL_1:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK-DAG: [[CMP_UB_NUM_IT:%.+]] = icmp sgt {{.+}}  [[OMP_UB_VAL_1]], 99
// CHECK: br {{.+}} [[CMP_UB_NUM_IT]], label %[[EUB_TRUE:.+]], label %[[EUB_FALSE:.+]]
// CHECK-DAG: [[EUB_TRUE]]:
// CHECK: br label %[[EUB_END:.+]]
// CHECK-DAG: [[EUB_FALSE]]:
// CHECK: [[OMP_UB_VAL2:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END]]
// CHECK-DAG: [[EUB_END]]:
// CHECK-DAG: [[EUB_RES:%.+]] = phi{{.+}} [ 99, %[[EUB_TRUE]] ], [ [[OMP_UB_VAL2]], %[[EUB_FALSE]] ]
// CHECK: store{{.+}} [[EUB_RES]], {{.+}}* [[OMP_UB]],

// initialize omp.iv
// CHECK: [[OMP_LB_VAL_1:%.+]] = load{{.+}}, {{.+}}* [[OMP_LB]],
// CHECK: store {{.+}} [[OMP_LB_VAL_1]], {{.+}}* [[OMP_IV]],

// check exit condition
// CHECK-DAG: [[OMP_IV_VAL_1:%.+]] = load {{.+}} [[OMP_IV]],
// CHECK: [[CMP_IV_UB:%.+]] = icmp slt {{.+}} [[OMP_IV_VAL_1]], 100
// CHECK: br {{.+}} [[CMP_IV_UB]], label %[[DIST_INNER_LOOP_BODY:.+]], label %[[DIST_INNER_LOOP_END:.+]]

// check that PrevLB and PrevUB are passed to the 'for'
// CHECK: [[DIST_INNER_LOOP_BODY]]:
// CHECK-DAG: [[OMP_PREV_LB:%.+]] = load {{.+}}, {{.+}} [[OMP_LB]],
// CHECK-64-DAG: [[OMP_PREV_LB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_LB]] to {{.+}}
// CHECK-DAG: [[OMP_PREV_UB:%.+]] = load {{.+}}, {{.+}} [[OMP_UB]],
// CHECK-64-DAG: [[OMP_PREV_UB_EXT:%.+]] = zext {{.+}} [[OMP_PREV_UB]] to {{.+}}

// check that distlb and distub are properly passed to the outlined function
// CHECK-32: {{call|invoke}} void [[OUTL4:@.+]]({{.*}} i32 [[OMP_PREV_LB]], i32 [[OMP_PREV_UB]]
// CHECK-64: {{call|invoke}} void [[OUTL4:@.+]]({{.*}} i64 [[OMP_PREV_LB_EXT]], i64 [[OMP_PREV_UB_EXT]]

// check DistInc
// CHECK-DAG: [[OMP_IV_VAL_3:%.+]] = load {{.+}}, {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_ST_VAL_1:%.+]] = load {{.+}}, {{.+}}* [[OMP_ST]],
// CHECK: [[OMP_IV_INC:%.+]] = add{{.+}} [[OMP_IV_VAL_3]], [[OMP_ST_VAL_1]]
// CHECK: store{{.+}} [[OMP_IV_INC]], {{.+}}* [[OMP_IV]],
// CHECK-DAG: [[OMP_LB_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
// CHECK-DAG: [[OMP_ST_VAL_2:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
// CHECK-DAG: [[OMP_LB_NEXT:%.+]] = add{{.+}} [[OMP_LB_VAL_2]], [[OMP_ST_VAL_2]]
// CHECK: store{{.+}} [[OMP_LB_NEXT]], {{.+}}* [[OMP_LB]],
// CHECK-DAG: [[OMP_UB_VAL_5:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
// CHECK-DAG: [[OMP_ST_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_ST]],
// CHECK-DAG: [[OMP_UB_NEXT:%.+]] = add{{.+}} [[OMP_UB_VAL_5]], [[OMP_ST_VAL_3]]
// CHECK: store{{.+}} [[OMP_UB_NEXT]], {{.+}}* [[OMP_UB]],

// Update UB
// CHECK-DAG: [[OMP_UB_VAL_6:%.+]] = load{{.+}}, {{.+}} [[OMP_UB]],
// CHECK-DAG: [[CMP_UB_NUM_IT_1:%.+]] = icmp sgt {{.+}}[[OMP_UB_VAL_6]], 99
// CHECK: br {{.+}} [[CMP_UB_NUM_IT_1]], label %[[EUB_TRUE_1:.+]], label %[[EUB_FALSE_1:.+]]
// CHECK-DAG: [[EUB_TRUE_1]]:
// CHECK: br label %[[EUB_END_1:.+]]
// CHECK-DAG: [[EUB_FALSE_1]]:
// CHECK: [[OMP_UB_VAL3:%.+]] = load{{.+}} [[OMP_UB]],
// CHECK: br label %[[EUB_END_1]]
// CHECK-DAG: [[EUB_END_1]]:
// CHECK-DAG: [[EUB_RES_1:%.+]] = phi{{.+}} [ 99, %[[EUB_TRUE_1]] ], [ [[OMP_UB_VAL3]], %[[EUB_FALSE_1]] ]
// CHECK: store{{.+}} [[EUB_RES_1]], {{.+}}* [[OMP_UB]],

// Store LB in IV
// CHECK-DAG: [[OMP_LB_VAL_3:%.+]] = load{{.+}}, {{.+}} [[OMP_LB]],
// CHECK: store{{.+}} [[OMP_LB_VAL_3]], {{.+}}* [[OMP_IV]],

// CHECK: [[DIST_INNER_LOOP_END]]:
// CHECK: call void @__kmpc_for_static_fini(


// CHECK-32: define internal void [[OUTL4]](
// CHECK-64: define internal void [[OUTL4]](
// CHECK: call void @__kmpc_for_static_init_4({{.+}}, {{.+}}, {{.+}} 33,
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: define weak void @__omp_offloading_{{.*}}_l74(i[[SZ:64|32]] %{{[^,]+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK: call void [[OUTLINED:@__omp_outlined.*]](i32* %{{.+}}, i32* %{{.+}}, i[[SZ]] %{{.*}}, i[[SZ]] %{{.*}}, i[[SZ]] %{{.*}}, [10 x [10 x i32]]* %{{.*}})
// CHECK: define internal void [[OUTLINED]](i32* noalias %{{.*}}, i32* noalias %{{.*}} i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, [10 x [10 x i32]]* nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK-DIV64: div i64
// CHECK-DIV32-NO: div i64

// CHECK: define weak void @__omp_offloading_{{.*}}_l81(i[[SZ:64|32]] %{{[^,]+}}, [1000 x i32]* nonnull align {{[0-9]+}} dereferenceable{{.*}}, i32* %{{[^)]+}})
// CHECK: call void [[OUTLINED:@__omp_outlined.*]](i32* %{{.+}}, i32* %{{.+}}, i[[SZ]] %{{.*}}, i[[SZ]] %{{.*}}, i[[SZ]] %{{.*}}, [1000 x i32]* %{{.*}}, i32* %{{.*}})
// CHECK: define internal void [[OUTLINED]](i32* noalias %{{.*}}, i32* noalias %{{.*}} i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, [1000 x i32]* nonnull align {{[0-9]+}} dereferenceable{{.*}}, i32* %{{.*}})

#endif
