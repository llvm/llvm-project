// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns -disable-O0-optnone | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix SEQ
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns -fopenmp-cuda-parallel-target-regions | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix PAR
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target if(0)
  {
    #pragma omp parallel
    {
      int a = 41;
    }
    a += 1;
  }

  #pragma omp target
  {
    #pragma omp parallel
    {
      int a = 42;
    }
    #pragma omp parallel if(0)
    {
      int a = 43;
    }
    #pragma omp parallel if(1)
    {
      int a = 44;
    }
    a += 1;
  }

  #pragma omp target if(n>40)
  {
    #pragma omp parallel if(n>1000)
    {
      int a = 45;
#pragma omp barrier
    }
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  #pragma omp target
  {
    #pragma omp parallel
    {
    #pragma omp critical
    ++a;
    }
    ++a;
  }
  return a;
}

int bar(int n){
  int a = 0;

  a += ftemplate<int>(n);

  return a;
}

// SEQ: [[MEM_TY:%.+]] = type { [128 x i8] }
// SEQ-DAG: [[SHARED_GLOBAL_RD:@.+]] = weak addrspace(3) global [[MEM_TY]] undef
// SEQ-DAG: [[KERNEL_PTR:@.+]] = internal addrspace(3) global i8* undef
// SEQ-DAG: [[KERNEL_SIZE:@.+]] = internal unnamed_addr constant i{{64|32}} 4
// SEQ-DAG: [[KERNEL_SHARED:@.+]] = internal unnamed_addr constant i16 1

// CHECK-NOT: define {{.*}}void {{@__omp_offloading_.+template.+l20}}_worker()

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l29}}_worker()
// CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
// CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
// CHECK: store i8* null, i8** [[OMP_WORK_FN]],
// CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
// CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
//
// CHECK: [[AWAIT_WORK]]
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]])
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
// CHECK: [[WF1:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
// CHECK: [[WM1:%.+]] = icmp eq i8* [[WF1]], bitcast (void (i16, i32)* [[PARALLEL_FN1:@.+]]_wrapper to i8*)
// CHECK: br i1 [[WM1]], label {{%?}}[[EXEC_PFN1:.+]], label {{%?}}[[CHECK_NEXT1:.+]]
//
// CHECK: [[EXEC_PFN1]]
// CHECK: call void [[PARALLEL_FN1]]_wrapper(
// CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
//
// CHECK: [[CHECK_NEXT1]]
// CHECK: [[WF2:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
// CHECK: [[WM2:%.+]] = icmp eq i8* [[WF2]], bitcast (void (i16, i32)* [[PARALLEL_FN2:@.+]]_wrapper to i8*)
// CHECK: br i1 [[WM2]], label {{%?}}[[EXEC_PFN2:.+]], label {{%?}}[[CHECK_NEXT2:.+]]
//
// CHECK: [[EXEC_PFN2]]
// CHECK: call void [[PARALLEL_FN2]]_wrapper(
// CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
//
// CHECK: [[CHECK_NEXT2]]
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

// CHECK: define {{.*}}void [[T6:@__omp_offloading_.+template.+l29]](i[[SZ:32|64]]
// Create local storage for each capture.
// CHECK:  [[LOCAL_A:%.+]] = alloca i[[SZ]],
// CHECK-DAG:  store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// Store captures in the context.
// CHECK-64-DAG:[[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
//
// CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
// CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
// CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
// CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
//
// CHECK: [[WORKER]]
// CHECK: {{call|invoke}} void [[T6]]_worker()
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
// CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* [[PARALLEL_FN1]]_wrapper to i8*))
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_serialized_parallel(
// CHECK: {{call|invoke}} void [[PARALLEL_FN3:@.+]](
// CHECK: call void @__kmpc_end_serialized_parallel(
// CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* [[PARALLEL_FN2]]_wrapper to i8*))
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK-64-DAG: load i32, i32* [[REF_A]]
// CHECK-32-DAG: load i32, i32* [[LOCAL_A]]
// CHECK: br label {{%?}}[[TERMINATE:.+]]
//
// CHECK: [[TERMINATE]]
// CHECK: call void @__kmpc_kernel_deinit(
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: br label {{%?}}[[EXIT]]
//
// CHECK: [[EXIT]]
// CHECK: ret void

// CHECK-DAG: define internal void [[PARALLEL_FN1]](
// CHECK: [[A:%.+]] = alloca i[[SZ:32|64]],
// CHECK: store i[[SZ]] 42, i[[SZ]]* %a,
// CHECK: ret void

// CHECK-DAG: define internal void [[PARALLEL_FN3]](
// CHECK: [[A:%.+]] = alloca i[[SZ:32|64]],
// CHECK: store i[[SZ]] 43, i[[SZ]]* %a,
// CHECK: ret void

// CHECK-DAG: define internal void [[PARALLEL_FN2]](
// CHECK: [[A:%.+]] = alloca i[[SZ:32|64]],
// CHECK: store i[[SZ]] 44, i[[SZ]]* %a,
// CHECK: ret void

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l46}}_worker()
// CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
// CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
// CHECK: store i8* null, i8** [[OMP_WORK_FN]],
// CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
// CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
//
// CHECK: [[AWAIT_WORK]]
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]])
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
// CHECK: [[WF:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
// CHECK: [[WM:%.+]] = icmp eq i8* [[WF]], bitcast (void (i16, i32)* [[PARALLEL_FN4:@.+]]_wrapper to i8*)
// CHECK: br i1 [[WM]], label {{%?}}[[EXEC_PFN:.+]], label {{%?}}[[CHECK_NEXT:.+]]
//
// CHECK: [[EXEC_PFN]]
// CHECK: call void [[PARALLEL_FN4]]_wrapper(
// CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
//
// CHECK: [[CHECK_NEXT]]
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

// CHECK: define {{.*}}void [[T6:@__omp_offloading_.+template.+l46]](i[[SZ:32|64]]
// Create local storage for each capture.
// CHECK:  [[LOCAL_N:%.+]] = alloca i[[SZ]],
// CHECK:  [[LOCAL_A:%.+]] = alloca i[[SZ]],
// CHECK:  [[LOCAL_AA:%.+]] = alloca i[[SZ]],
// CHECK:  [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK-DAG:  store i[[SZ]] [[ARG_N:%.+]], i[[SZ]]* [[LOCAL_N]]
// CHECK-DAG:  store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:  store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[REF_N:%.+]] = bitcast i[[SZ]]* [[LOCAL_N]] to i32*
// CHECK-64-DAG:[[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[REF_AA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
//
// CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
// CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
// CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
// CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
//
// CHECK: [[WORKER]]
// CHECK: {{call|invoke}} void [[T6]]_worker()
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
// CHECK-64: [[N:%.+]] = load i32, i32* [[REF_N]],
// CHECK-32: [[N:%.+]] = load i32, i32* [[LOCAL_N]],
// CHECK: [[CMP:%.+]] = icmp sgt i32 [[N]], 1000
// CHECK: br i1 [[CMP]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
//
// CHECK: [[IF_THEN]]
// CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* [[PARALLEL_FN4]]_wrapper to i8*))
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: br label {{%?}}[[IF_END:.+]]
//
// CHECK: [[IF_ELSE]]
// CHECK: call void @__kmpc_serialized_parallel(
// CHECK: {{call|invoke}} void [[PARALLEL_FN4]](
// CHECK: call void @__kmpc_end_serialized_parallel(
// br label [[IF_END]]
//
// CHECK: [[IF_END]]
// CHECK-64-DAG: load i32, i32* [[REF_A]]
// CHECK-32-DAG: load i32, i32* [[LOCAL_A]]
// CHECK-DAG:    load i16, i16* [[REF_AA]]
// CHECK-DAG:    getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
//
// CHECK: br label {{%?}}[[TERMINATE:.+]]
//
// CHECK: [[TERMINATE]]
// CHECK: call void @__kmpc_kernel_deinit(
// CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
// CHECK: br label {{%?}}[[EXIT]]
//
// CHECK: [[EXIT]]
// CHECK: ret void

// CHECK: noinline
// CHECK-NEXT: define internal void [[PARALLEL_FN4]](
// CHECK: [[A:%.+]] = alloca i[[SZ:32|64]],
// CHECK: store i[[SZ]] 45, i[[SZ]]* %a,
// CHECK: call void @__kmpc_barrier(%struct.ident_t* @{{.+}}, i32 %{{.+}})
// CHECK: ret void

// CHECK: declare void @__kmpc_barrier(%struct.ident_t*, i32) #[[#CONVERGENT:]]

// CHECK: Function Attrs: convergent noinline norecurse nounwind
// CHECK-NEXT: [[PARALLEL_FN4]]_wrapper

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l58}}_worker()
// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l58}}(
// CHECK-32: [[A_ADDR:%.+]] = alloca i32,
// CHECK-64: [[A_ADDR:%.+]] = alloca i64,
// CHECK-64: [[CONV:%.+]] = bitcast i64* [[A_ADDR]] to i32*
// SEQ: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// SEQ: [[SIZE:%.+]] = load i{{64|32}}, i{{64|32}}* [[KERNEL_SIZE]],
// SEQ: call void @__kmpc_get_team_static_memory(i16 0, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* [[SHARED_GLOBAL_RD]], i32 0, i32 0, i32 0) to i8*), i{{64|32}} [[SIZE]], i16 [[IS_SHARED]], i8** addrspacecast (i8* addrspace(3)* [[KERNEL_PTR]] to i8**))
// SEQ: [[KERNEL_RD:%.+]] = load i8*, i8* addrspace(3)* [[KERNEL_PTR]],
// SEQ: [[STACK:%.+]] = getelementptr inbounds i8, i8* [[KERNEL_RD]], i{{64|32}} 0
// PAR: [[STACK:%.+]] = call i8* @__kmpc_data_sharing_push_stack(i{{32|64}} 4, i16 1)
// CHECK: [[BC:%.+]] = bitcast i8* [[STACK]] to %struct._globalized_locals_ty*
// CHECK-32: [[A:%.+]] = load i32, i32* [[A_ADDR]],
// CHECK-64: [[A:%.+]] = load i32, i32* [[CONV]],
// CHECK: [[GLOBAL_A_ADDR:%.+]] = getelementptr inbounds %struct._globalized_locals_ty, %struct._globalized_locals_ty* [[BC]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: store i32 [[A]], i32* [[GLOBAL_A_ADDR]],
// SEQ: [[IS_SHARED:%.+]] = load i16, i16* [[KERNEL_SHARED]],
// SEQ: call void @__kmpc_restore_team_static_memory(i16 0, i16 [[IS_SHARED]])
// PAR: call void @__kmpc_data_sharing_pop_stack(i8* [[STACK]])

// CHECK-LABEL: define internal void @{{.+}}(i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK:  [[CC:%.+]] = alloca i32,
// CHECK:  [[MASK:%.+]] = call i32 @__kmpc_warp_active_thread_mask(){{$}}
// CHECK:  [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK:  [[NUM_THREADS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
// CHECK:  store i32 0, i32* [[CC]],
// CHECK:  br label

// CHECK:  [[CC_VAL:%.+]] = load i32, i32* [[CC]],
// CHECK:  [[RES:%.+]] = icmp slt i32 [[CC_VAL]], [[NUM_THREADS]]
// CHECK:  br i1 [[RES]], label

// CHECK:  [[CC_VAL:%.+]] = load i32, i32* [[CC]],
// CHECK:  [[RES:%.+]] = icmp eq i32 [[TID]], [[CC_VAL]]
// CHECK:  br i1 [[RES]], label

// CHECK:  call void @__kmpc_critical(
// CHECK:  load i32, i32*
// CHECK:  add nsw i32
// CHECK:  store i32
// CHECK:  call void @__kmpc_end_critical(

// CHECK:  call void @__kmpc_syncwarp(i32 [[MASK]]){{$}}
// CHECK:  [[NEW_CC_VAL:%.+]] = add nsw i32 [[CC_VAL]], 1
// CHECK:  store i32 [[NEW_CC_VAL]], i32* [[CC]],
// CHECK:  br label

// CHECK: declare i32 @__kmpc_warp_active_thread_mask() #[[#CONVERGENT:]]
// CHECK: declare void @__kmpc_syncwarp(i32) #[[#CONVERGENT:]]

// CHECK: attributes #[[#CONVERGENT:]] = {{.*}} convergent {{.*}}

#endif
