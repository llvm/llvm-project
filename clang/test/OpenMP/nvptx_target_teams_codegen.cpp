// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check that the execution mode of all 2 target regions is set to Generic Mode.
// CHECK-DAG: {{@__omp_offloading_.+l27}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l32}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l37}}_exec_mode = weak constant i8 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target teams if(0)
  {
    b[2] += 1;
  }

  #pragma omp target teams if(1)
  {
    a = '1';
  }

  #pragma omp target teams if(n>40)
  {
    aa = 1;
  }

  #pragma omp target teams
  {
#pragma omp parallel
#pragma omp parallel
    aa = 1;
  }

  return a;
}

int bar(int n){
  int a = 0;

  a += ftemplate<char>(n);

  return a;
}

  // CHECK-NOT: define {{.*}}void {{@__omp_offloading_.+template.+l22}}_worker()






  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l27}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]], i16 1)
  // CHECK: [[KPRB:%.+]] = zext i1 [[KPR]] to i8
  // store i8 [[KPRB]], i8* [[OMP_EXEC_STATUS]], align 1
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]]
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: call void @__kmpc_kernel_end_parallel()
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T1:@__omp_offloading_.+template.+l27]](i[[SZ:32|64]] [[A:%[^)]+]])
  // CHECK: store i[[SZ]] [[A]], i[[SZ]]* [[A_ADDR:%.+]], align
  // CHECK: [[CONV:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i8*

  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void [[T1]]_worker()
  // CHECK: br label {{%?}}[[EXIT:.+]]
  //
  // CHECK: [[CHECK_MASTER]]
  // CHECK-DAG: [[CMTID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[CMNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[CMWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[IS_MASTER:%.+]] = icmp eq i32 [[CMTID]],
  // CHECK: br i1 [[IS_MASTER]], label {{%?}}[[MASTER:.+]], label {{%?}}[[EXIT]]
  //
  // CHECK: [[MASTER]]
  // CHECK-DAG: [[MNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[MWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[MTMP1:%.+]] = sub nuw i32 [[MNTH]], [[MWS]]
  // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]
  //
  // CHECK-NOT: kmpc_fork_teams
  // CHECK: [[A_VAL:%.+]] = load i8, i8* [[CONV]], align
  // CHECK: [[ACP:%.+]] = bitcast i[[SZ]]* [[AC:%.+]] to i8*
  // CHECK: store i8 [[A_VAL]], i8* [[ACP]], align
  // CHECK: [[ACV:%.+]] = load i[[SZ]], i[[SZ]]* [[AC]], align
  // CHECK: call void [[PARALLEL:@.+]](i32* %{{.+}}, i32* %{{.+}}, i[[SZ]] [[ACV]])
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define internal void [[PARALLEL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i[[SZ]] [[A_VAL:%.+]])
  // CHECK: [[A_ADDR:%.+]] = alloca i[[SZ]],
  // CHECK: store i[[SZ]] [[A_VAL]], i[[SZ]]* [[A_ADDR]],
  // CHECK: [[CONV:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i8*
  // CHECK: store i8 49, i8* [[CONV]],
  // CHECK: ret void

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l32}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]], i16 1)
  // CHECK: [[KPRB:%.+]] = zext i1 [[KPR]] to i8
  // store i8 [[KPRB]], i8* [[OMP_EXEC_STATUS]], align 1
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]]
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: call void @__kmpc_kernel_end_parallel()
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T2:@__omp_offloading_.+template.+l32]](i[[SZ:32|64]] [[AA:%[^)]+]])
  // CHECK: store i[[SZ]] [[AA]], i[[SZ]]* [[AA_ADDR:%.+]], align
  // CHECK: [[CONV:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*

  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void [[T2]]_worker()
  // CHECK: br label {{%?}}[[EXIT:.+]]
  //
  // CHECK: [[CHECK_MASTER]]
  // CHECK-DAG: [[CMTID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[CMNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[CMWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[IS_MASTER:%.+]] = icmp eq i32 [[CMTID]],
  // CHECK: br i1 [[IS_MASTER]], label {{%?}}[[MASTER:.+]], label {{%?}}[[EXIT]]
  //
  // CHECK: [[MASTER]]
  // CHECK-DAG: [[MNTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[MWS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[MTMP1:%.+]] = sub nuw i32 [[MNTH]], [[MWS]]
  // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]
  //
  // CHECK-NOT: kmpc_fork_teams
  // CHECK: [[AA_VAL:%.+]] = load i16, i16* [[CONV]], align
  // CHECK: [[ACP:%.+]] = bitcast i[[SZ]]* [[AC:%.+]] to i16*
  // CHECK: store i16 [[AA_VAL]], i16* [[ACP]], align
  // CHECK: [[ACV:%.+]] = load i[[SZ]], i[[SZ]]* [[AC]], align
  // CHECK: call void [[PARALLEL:@.+]](i32* %{{.+}}, i32* %{{.+}}, i[[SZ]] [[ACV]])
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define internal void [[PARALLEL]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i[[SZ]] [[A_VAL:%.+]])
  // CHECK: [[A_ADDR:%.+]] = alloca i[[SZ]],
  // CHECK: store i[[SZ]] [[A_VAL]], i[[SZ]]* [[A_ADDR]],
  // CHECK: [[CONV:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i16*
  // CHECK: store i16 1, i16* [[CONV]],
  // CHECK: ret void

// CHECK: define weak void @__omp_offloading_{{.*}}ftemplate{{.*}}_l37(
// CHECK: call void @__kmpc_spmd_kernel_init(i32 {{.+}}, i16 1, i16 0)
// CHECK: call void @__kmpc_data_sharing_init_stack_spmd
// CHECK-NOT: call i8* @__kmpc_data_sharing_push_stack(
// CHECK-NOT: call void @__kmpc_serialized_parallel(
// CHECK: call void [[L0:@.+]](i32* %{{.+}}, i32* %{{.+}}, i[[SZ]] %{{.+}})
// CHECK-NOT: call void @__kmpc_end_serialized_parallel(
// CHECK-NOT: call void @__kmpc_data_sharing_pop_stack(
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
// CHECK: ret

// CHECK: define internal void [[L0]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i[[SZ]] %{{.+}})
// CHECK: call void [[L1:@.+]](i32* %{{.+}}, i32* %{{.+}}, i16* %{{.+}})
// CHECK: ret void

// CHECK: define internal void [[L1]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i16* dereferenceable
// CHECK: call void @__kmpc_serialized_parallel(
// CHECK: call void [[L2:@.+]](i32* %{{.+}}, i32* %{{.+}}, i16* %{{.+}})
// CHECK: call void @__kmpc_end_serialized_parallel(
// CHECK: ret void

// CHECK: define internal void [[L2]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i16* dereferenceable
// CHECK: store i16 1, i16* %
// CHECK: ret void

#endif
