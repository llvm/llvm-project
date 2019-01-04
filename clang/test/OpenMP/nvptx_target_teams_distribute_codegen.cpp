// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[MEM_TY:%.+]] = type { [128 x i8] }
// CHECK-DAG: {{@__omp_offloading_.+}}_l19_exec_mode = weak constant i8 1
// CHECK-DAG: internal unnamed_addr constant i{{64|32}} 4

template<typename tx>
tx ftemplate(int n) {
  int i;

  #pragma omp target teams distribute
  for (i = 0; i < 10; ++i)
  {
#pragma omp parallel
    ++i;
  }

  return i;
}

int bar(int n){
  int a = 0;

  a += ftemplate<char>(n);

  return a;
}

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l19}}_worker()
  // CHECK: ret void

  // CHECK: define {{.*}}void {{@__omp_offloading_.+template.+l19}}()

  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void {{@__omp_offloading_.+template.+l19}}_worker()
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
  // CHECK: call void @__kmpc_get_team_static_memory(i16 0, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([[MEM_TY]], [[MEM_TY]] addrspace(3)* @{{.+}}, i32 0, i32 0, i32 0) to i8*), i{{64|32}} 4, i16 1, i8** addrspacecast (i8* addrspace(3)* [[BUF:@.+]] to i8**))
  // CHECK: [[PTR:%.+]] = load i8*, i8* addrspace(3)* [[BUF]],
  // CHECK: [[RD:%.+]] = bitcast i8* [[PTR]] to [[GLOB_TY:%.+]]*
  // CHECK: [[I_ADDR:%.+]] = getelementptr inbounds [[GLOB_TY]], [[GLOB_TY]]* [[RD]], i32 0, i32 0
  //
  // CHECK: call void @__kmpc_for_static_init_4(
  // CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @{{.+}} to i8*), i16 1)
  // CHECK: call void @__kmpc_begin_sharing_variables(i8*** [[SHARED_VARS_PTR:%.+]], i{{64|32}} 1)
  // CHECK: [[SHARED_VARS_BUF:%.+]] = load i8**, i8*** [[SHARED_VARS_PTR]],
  // CHECK: [[I_ADDR_BC:%.+]] = bitcast i32* [[I_ADDR]] to i8*
  // CHECK: store i8* [[I_ADDR_BC]], i8** [[SHARED_VARS_BUF]],
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: call void @__kmpc_end_sharing_variables()
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

#endif
