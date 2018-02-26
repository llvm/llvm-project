// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
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
    }
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

  // CHECK-NOT: define {{.*}}void {{@__omp_offloading_.+template.+l17}}_worker()






  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l26}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]],
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
  // CHECK: [[WM1:%.+]] = icmp eq i8* [[WF1]], bitcast (void (i16, i32, i8**)* [[PARALLEL_FN1:@.+]]_wrapper to i8*)
  // CHECK: br i1 [[WM1]], label {{%?}}[[EXEC_PFN1:.+]], label {{%?}}[[CHECK_NEXT1:.+]]
  //
  // CHECK: [[EXEC_PFN1]]
  // CHECK: call void [[PARALLEL_FN1]]_wrapper(
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[CHECK_NEXT1]]
  // CHECK: [[WF2:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[WM2:%.+]] = icmp eq i8* [[WF2]], bitcast (void (i16, i32, i8**)* [[PARALLEL_FN2:@.+]]_wrapper to i8*)
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
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T6:@__omp_offloading_.+template.+l26]](i[[SZ:32|64]]
  // Create local storage for each capture.
  // CHECK:  [[LOCAL_A:%.+]] = alloca i[[SZ]],
  // CHECK-DAG:  store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
  // Store captures in the context.
  // CHECK-64-DAG:[[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
  //
  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub i32 [[NTH]], [[WS]]
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
  // CHECK: [[MTMP1:%.+]] = sub i32 [[MNTH]], [[MWS]]
  // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]
  // CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32, i8**)* [[PARALLEL_FN1]]_wrapper to i8*),
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: call void @__kmpc_serialized_parallel(
  // CHECK: {{call|invoke}} void [[PARALLEL_FN3:@.+]](
  // CHECK: call void @__kmpc_end_serialized_parallel(
  // CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32, i8**)* [[PARALLEL_FN2]]_wrapper to i8*),
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK-64-DAG: load i32, i32* [[REF_A]]
  // CHECK-32-DAG: load i32, i32* [[LOCAL_A]]
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @llvm.nvvm.barrier0()
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







  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l43}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: [[KPR:%.+]] = call i1 @__kmpc_kernel_parallel(i8** [[OMP_WORK_FN]],
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
  // CHECK: [[WM:%.+]] = icmp eq i8* [[WF]], bitcast (void (i16, i32, i8**)* [[PARALLEL_FN4:@.+]]_wrapper to i8*)
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
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T6:@__omp_offloading_.+template.+l43]](i[[SZ:32|64]]
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
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub i32 [[NTH]], [[WS]]
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
  // CHECK: [[MTMP1:%.+]] = sub i32 [[MNTH]], [[MWS]]
  // CHECK: call void @__kmpc_kernel_init(i32 [[MTMP1]]
  // CHECK-64: [[N:%.+]] = load i32, i32* [[REF_N]],
  // CHECK-32: [[N:%.+]] = load i32, i32* [[LOCAL_N]],
  // CHECK: [[CMP:%.+]] = icmp sgt i32 [[N]], 1000
  // CHECK: br i1 [[CMP]], label {{%?}}[[IF_THEN:.+]], label {{%?}}[[IF_ELSE:.+]]
  //
  // CHECK: [[IF_THEN]]
  // CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32, i8**)* [[PARALLEL_FN4]]_wrapper to i8*),
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: call void @llvm.nvvm.barrier0()
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
  // CHECK: call void @llvm.nvvm.barrier0()
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define internal void [[PARALLEL_FN4]](
  // CHECK: [[A:%.+]] = alloca i[[SZ:32|64]],
  // CHECK: store i[[SZ]] 45, i[[SZ]]* %a,
  // CHECK: ret void
#endif
