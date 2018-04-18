// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-cuda-mode -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Check for the data transfer medium in shared memory to transfer the reduction list to the first warp.
// CHECK-DAG: [[TRANSFER_STORAGE:@.+]] = common addrspace([[SHARED_ADDRSPACE:[0-9]+]]) global [32 x i64]

// Check that the execution mode of all 3 target regions is set to Spmd Mode.
// CHECK-DAG: {{@__omp_offloading_.+l27}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l32}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l38}}_exec_mode = weak constant i8 0

template<typename tx>
tx ftemplate(int n) {
  int a;
  short b;
  tx c;
  float d;
  double e;

  #pragma omp target parallel reduction(+: e)
  {
    e += 5;
  }

  #pragma omp target parallel reduction(^: c) reduction(*: d)
  {
    c ^= 2;
    d *= 33;
  }

  #pragma omp target parallel reduction(|: a) reduction(max: b)
  {
    a |= 1;
    b = 99 > b ? 99 : b;
  }

  return a+b+c+d+e;
}

int bar(int n){
  int a = 0;

  a += ftemplate<char>(n);

  return a;
}

  // CHECK: define {{.*}}void {{@__omp_offloading_.+template.+l27}}(
  //
  // CHECK: call void @__kmpc_spmd_kernel_init(
  // CHECK: br label {{%?}}[[EXECUTE:.+]]
  //
  // CHECK: [[EXECUTE]]
  // CHECK: {{call|invoke}} void [[PFN:@.+]](i32*
  // CHECK: call void @__kmpc_spmd_kernel_deinit()
  //
  //
  // define internal void [[PFN]](
  // CHECK: store double {{[0\.e\+]+}}, double* [[E:%.+]], align
  // CHECK: [[EV:%.+]] = load double, double* [[E]], align
  // CHECK: [[ADD:%.+]] = fadd double [[EV]], 5
  // CHECK: store double [[ADD]], double* [[E]], align
  // CHECK: [[PTR1:%.+]] = getelementptr inbounds [[RLT:.+]], [1 x i8*]* [[RL:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[E_CAST:%.+]] = bitcast double* [[E]] to i8*
  // CHECK: store i8* [[E_CAST]], i8** [[PTR1]], align
  // CHECK: [[ARG_RL:%.+]] = bitcast [[RLT]]* [[RL]] to i8*
  // CHECK: [[RET:%.+]] = call i32 @__kmpc_nvptx_parallel_reduce_nowait(i32 {{.+}}, i32 1, i{{32|64}} {{4|8}}, i8* [[ARG_RL]], void (i8*, i16, i16, i16)* [[SHUFFLE_REDUCE_FN:@.+]], void (i8*, i32)* [[WARP_COPY_FN:@.+]])
  // CHECK: switch i32 [[RET]], label {{%?}}[[DEFAULTLABEL:.+]] [
  // CHECK: i32 1, label {{%?}}[[REDLABEL:.+]]

  // CHECK: [[REDLABEL]]
  // CHECK: [[E_INV:%.+]] = load double, double* [[E_IN:%.+]], align
  // CHECK: [[EV:%.+]] = load double, double* [[E]], align
  // CHECK: [[ADD:%.+]] = fadd double [[E_INV]], [[EV]]
  // CHECK: store double [[ADD]], double* [[E_IN]], align
  // CHECK: call void @__kmpc_nvptx_end_reduce_nowait(
  // CHECK: br label %[[DEFAULTLABEL]]
  //
  // CHECK: [[DEFAULTLABEL]]
  // CHECK: ret

  //
  // Reduction function
  // CHECK: define internal void [[REDUCTION_FUNC:@.+]](i8*, i8*)
  // CHECK: [[VAR_RHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_RHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR_RHS_VOID:%.+]] = load i8*, i8** [[VAR_RHS_REF]],
  // CHECK: [[VAR_RHS:%.+]] = bitcast i8* [[VAR_RHS_VOID]] to double*
  //
  // CHECK: [[VAR_LHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_LHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR_LHS_VOID:%.+]] = load i8*, i8** [[VAR_LHS_REF]],
  // CHECK: [[VAR_LHS:%.+]] = bitcast i8* [[VAR_LHS_VOID]] to double*
  //
  // CHECK: [[VAR_LHS_VAL:%.+]] = load double, double* [[VAR_LHS]],
  // CHECK: [[VAR_RHS_VAL:%.+]] = load double, double* [[VAR_RHS]],
  // CHECK: [[RES:%.+]] = fadd double [[VAR_LHS_VAL]], [[VAR_RHS_VAL]]
  // CHECK: store double [[RES]], double* [[VAR_LHS]],
  // CHECK: ret void

  //
  // Shuffle and reduce function
  // CHECK: define internal void [[SHUFFLE_REDUCE_FN]](i8*, i16 {{.*}}, i16 {{.*}}, i16 {{.*}})
  // CHECK: [[REMOTE_RED_LIST:%.+]] = alloca [[RLT]], align
  // CHECK: [[REMOTE_ELT:%.+]] = alloca double
  //
  // CHECK: [[LANEID:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[LANEOFFSET:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[ALGVER:%.+]] = load i16, i16* {{.+}}, align
  //
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to double*
  // CHECK: [[ELT_VAL:%.+]] = load double, double* [[ELT]], align
  //
  // CHECK: [[ELT_CAST:%.+]] = bitcast double [[ELT_VAL]] to i64
  // CHECK: [[WS32:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[WS:%.+]] = trunc i32 [[WS32]] to i16
  // CHECK: [[REMOTE_ELT_VAL64:%.+]] = call i64 @__kmpc_shuffle_int64(i64 [[ELT_CAST]], i16 [[LANEOFFSET]], i16 [[WS]])
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = bitcast i64 [[REMOTE_ELT_VAL64]] to double
  //
  // CHECK: store double [[REMOTE_ELT_VAL]], double* [[REMOTE_ELT]], align
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = bitcast double* [[REMOTE_ELT]] to i8*
  // CHECK: store i8* [[REMOTE_ELT_VOID]], i8** [[REMOTE_ELT_REF]], align
  //
  // Condition to reduce
  // CHECK: [[CONDALG0:%.+]] = icmp eq i16 [[ALGVER]], 0
  //
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp ult i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[CONDALG1:%.+]] = and i1 [[COND1]], [[COND2]]
  //
  // CHECK: [[COND3:%.+]] = icmp eq i16 [[ALGVER]], 2
  // CHECK: [[COND4:%.+]] = and i16 [[LANEID]], 1
  // CHECK: [[COND5:%.+]] = icmp eq i16 [[COND4]], 0
  // CHECK: [[COND6:%.+]] = and i1 [[COND3]], [[COND5]]
  // CHECK: [[COND7:%.+]] = icmp sgt i16 [[LANEOFFSET]], 0
  // CHECK: [[CONDALG2:%.+]] = and i1 [[COND6]], [[COND7]]
  //
  // CHECK: [[COND8:%.+]] = or i1 [[CONDALG0]], [[CONDALG1]]
  // CHECK: [[SHOULD_REDUCE:%.+]] = or i1 [[COND8]], [[CONDALG2]]
  // CHECK: br i1 [[SHOULD_REDUCE]], label {{%?}}[[DO_REDUCE:.+]], label {{%?}}[[REDUCE_ELSE:.+]]
  //
  // CHECK: [[DO_REDUCE]]
  // CHECK: [[RED_LIST1_VOID:%.+]] = bitcast [[RLT]]* [[RED_LIST]] to i8*
  // CHECK: [[RED_LIST2_VOID:%.+]] = bitcast [[RLT]]* [[REMOTE_RED_LIST]] to i8*
  // CHECK: call void [[REDUCTION_FUNC]](i8* [[RED_LIST1_VOID]], i8* [[RED_LIST2_VOID]])
  // CHECK: br label {{%?}}[[REDUCE_CONT:.+]]
  //
  // CHECK: [[REDUCE_ELSE]]
  // CHECK: br label {{%?}}[[REDUCE_CONT]]
  //
  // CHECK: [[REDUCE_CONT]]
  // Now check if we should just copy over the remote reduction list
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp uge i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[SHOULD_COPY:%.+]] = and i1 [[COND1]], [[COND2]]
  // CHECK: br i1 [[SHOULD_COPY]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // CHECK: [[DO_COPY]]
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = load i8*, i8** [[REMOTE_ELT_REF]],
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT:%.+]] = bitcast i8* [[REMOTE_ELT_VOID]] to double*
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = load double, double* [[REMOTE_ELT]], align
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to double*
  // CHECK: store double [[REMOTE_ELT_VAL]], double* [[ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // CHECK: [[COPY_CONT]]
  // CHECK: void

  //
  // Inter warp copy function
  // CHECK: define internal void [[WARP_COPY_FN]](i8*, i32)
  // CHECK-DAG: [[LANEID:%.+]] = and i32 {{.+}}, 31
  // CHECK-DAG: [[WARPID:%.+]] = ashr i32 {{.+}}, 5
  // CHECK-DAG: [[RED_LIST:%.+]] = bitcast i8* {{.+}} to [[RLT]]*
  // CHECK: [[IS_WARP_MASTER:%.+]] = icmp eq i32 [[LANEID]], 0
  // CHECK: br i1 [[IS_WARP_MASTER]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // [[DO_COPY]]
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to double*
  // CHECK: [[ELT_VAL:%.+]] = load double, double* [[ELT]], align
  //
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[WARPID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to double addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: store double [[ELT_VAL]], double addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // Barrier after copy to shared memory storage medium.
  // CHECK: [[COPY_CONT]]
  // CHECK: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[ACTIVE_THREADS:%.+]] = mul nsw i32 [[ACTIVE_WARPS:%.+]], [[WS]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  //
  // Read into warp 0.
  // CHECK: [[IS_W0_ACTIVE_THREAD:%.+]] = icmp ult i32 [[TID:%.+]], [[ACTIVE_WARPS]]
  // CHECK: br i1 [[IS_W0_ACTIVE_THREAD]], label {{%?}}[[DO_READ:.+]], label {{%?}}[[READ_ELSE:.+]]
  //
  // CHECK: [[DO_READ]]
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[TID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to double addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: [[MEDIUM_ELT_VAL:%.+]] = load double, double addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to double*
  // CHECK: store double [[MEDIUM_ELT_VAL]], double* [[ELT]], align
  // CHECK: br label {{%?}}[[READ_CONT:.+]]
  //
  // CHECK: [[READ_ELSE]]
  // CHECK: br label {{%?}}[[READ_CONT]]
  //
  // CHECK: [[READ_CONT]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  // CHECK: ret










  // CHECK: define {{.*}}void {{@__omp_offloading_.+template.+l32}}(
  //
  // CHECK: call void @__kmpc_spmd_kernel_init(
  // CHECK: br label {{%?}}[[EXECUTE:.+]]
  //
  // CHECK: [[EXECUTE]]
  // CHECK: {{call|invoke}} void [[PFN1:@.+]](i32*
  // CHECK: call void @__kmpc_spmd_kernel_deinit()
  //
  //
  // define internal void [[PFN1]](
  // CHECK: store float {{1\.[0e\+]+}}, float* [[D:%.+]], align
  // CHECK: [[C_VAL:%.+]] = load i8, i8* [[C:%.+]], align
  // CHECK: [[CONV:%.+]] = sext i8 [[C_VAL]] to i32
  // CHECK: [[XOR:%.+]] = xor i32 [[CONV]], 2
  // CHECK: [[TRUNC:%.+]] = trunc i32 [[XOR]] to i8
  // CHECK: store i8 [[TRUNC]], i8* [[C]], align
  // CHECK: [[DV:%.+]] = load float, float* [[D]], align
  // CHECK: [[MUL:%.+]] = fmul float [[DV]], {{[0-9e\.\+]+}}
  // CHECK: store float [[MUL]], float* [[D]], align
  // CHECK: [[PTR1:%.+]] = getelementptr inbounds [[RLT:.+]], [2 x i8*]* [[RL:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: store i8* [[C]], i8** [[PTR1]], align
  // CHECK: [[PTR2:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RL]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[D_CAST:%.+]] = bitcast float* [[D]] to i8*
  // CHECK: store i8* [[D_CAST]], i8** [[PTR2]], align
  // CHECK: [[ARG_RL:%.+]] = bitcast [[RLT]]* [[RL]] to i8*
  // CHECK: [[RET:%.+]] = call i32 @__kmpc_nvptx_parallel_reduce_nowait(i32 {{.+}}, i32 2, i{{32|64}} {{8|16}}, i8* [[ARG_RL]], void (i8*, i16, i16, i16)* [[SHUFFLE_REDUCE_FN:@.+]], void (i8*, i32)* [[WARP_COPY_FN:@.+]])
  // CHECK: switch i32 [[RET]], label {{%?}}[[DEFAULTLABEL:.+]] [
  // CHECK: i32 1, label {{%?}}[[REDLABEL:.+]]

  // CHECK: [[REDLABEL]]
  // CHECK: [[C_INV8:%.+]] = load i8, i8* [[C_IN:%.+]], align
  // CHECK: [[C_INV:%.+]] = sext i8 [[C_INV8]] to i32
  // CHECK: [[CV8:%.+]] = load i8, i8* [[C]], align
  // CHECK: [[CV:%.+]] = sext i8 [[CV8]] to i32
  // CHECK: [[XOR:%.+]] = xor i32 [[C_INV]], [[CV]]
  // CHECK: [[TRUNC:%.+]] = trunc i32 [[XOR]] to i8
  // CHECK: store i8 [[TRUNC]], i8* [[C_IN]], align
  // CHECK: [[D_INV:%.+]] = load float, float* [[D_IN:%.+]], align
  // CHECK: [[DV:%.+]] = load float, float* [[D]], align
  // CHECK: [[MUL:%.+]] = fmul float [[D_INV]], [[DV]]
  // CHECK: store float [[MUL]], float* [[D_IN]], align
  // CHECK: call void @__kmpc_nvptx_end_reduce_nowait(
  // CHECK: br label %[[DEFAULTLABEL]]
  //
  // CHECK: [[DEFAULTLABEL]]
  // CHECK: ret

  //
  // Reduction function
  // CHECK: define internal void [[REDUCTION_FUNC:@.+]](i8*, i8*)
  // CHECK: [[VAR1_RHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_RHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR1_RHS:%.+]] = load i8*, i8** [[VAR1_RHS_REF]],
  //
  // CHECK: [[VAR1_LHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_LHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR1_LHS:%.+]] = load i8*, i8** [[VAR1_LHS_REF]],
  //
  // CHECK: [[VAR2_RHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_RHS]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[VAR2_RHS_VOID:%.+]] = load i8*, i8** [[VAR2_RHS_REF]],
  // CHECK: [[VAR2_RHS:%.+]] = bitcast i8* [[VAR2_RHS_VOID]] to float*
  //
  // CHECK: [[VAR2_LHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_LHS]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[VAR2_LHS_VOID:%.+]] = load i8*, i8** [[VAR2_LHS_REF]],
  // CHECK: [[VAR2_LHS:%.+]] = bitcast i8* [[VAR2_LHS_VOID]] to float*
  //
  // CHECK: [[VAR1_LHS_VAL8:%.+]] = load i8, i8* [[VAR1_LHS]],
  // CHECK: [[VAR1_LHS_VAL:%.+]] = sext i8 [[VAR1_LHS_VAL8]] to i32
  // CHECK: [[VAR1_RHS_VAL8:%.+]] = load i8, i8* [[VAR1_RHS]],
  // CHECK: [[VAR1_RHS_VAL:%.+]] = sext i8 [[VAR1_RHS_VAL8]] to i32
  // CHECK: [[XOR:%.+]] = xor i32 [[VAR1_LHS_VAL]], [[VAR1_RHS_VAL]]
  // CHECK: [[RES:%.+]] = trunc i32 [[XOR]] to i8
  // CHECK: store i8 [[RES]], i8* [[VAR1_LHS]],
  //
  // CHECK: [[VAR2_LHS_VAL:%.+]] = load float, float* [[VAR2_LHS]],
  // CHECK: [[VAR2_RHS_VAL:%.+]] = load float, float* [[VAR2_RHS]],
  // CHECK: [[RES:%.+]] = fmul float [[VAR2_LHS_VAL]], [[VAR2_RHS_VAL]]
  // CHECK: store float [[RES]], float* [[VAR2_LHS]],
  // CHECK: ret void

  //
  // Shuffle and reduce function
  // CHECK: define internal void [[SHUFFLE_REDUCE_FN]](i8*, i16 {{.*}}, i16 {{.*}}, i16 {{.*}})
  // CHECK: [[REMOTE_RED_LIST:%.+]] = alloca [[RLT]], align
  // CHECK: [[REMOTE_ELT1:%.+]] = alloca i8
  // CHECK: [[REMOTE_ELT2:%.+]] = alloca float
  //
  // CHECK: [[LANEID:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[LANEOFFSET:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[ALGVER:%.+]] = load i16, i16* {{.+}}, align
  //
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VAL:%.+]] = load i8, i8* [[ELT_VOID]], align
  //
  // CHECK: [[ELT_CAST:%.+]] = sext i8 [[ELT_VAL]] to i32
  // CHECK: [[WS32:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[WS:%.+]] = trunc i32 [[WS32]] to i16
  // CHECK: [[REMOTE_ELT1_VAL32:%.+]] = call i32 @__kmpc_shuffle_int32(i32 [[ELT_CAST]], i16 [[LANEOFFSET]], i16 [[WS]])
  // CHECK: [[REMOTE_ELT1_VAL:%.+]] = trunc i32 [[REMOTE_ELT1_VAL32]] to i8
  //
  // CHECK: store i8 [[REMOTE_ELT1_VAL]], i8* [[REMOTE_ELT1]], align
  // CHECK: store i8* [[REMOTE_ELT1]], i8** [[REMOTE_ELT_REF]], align
  //
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to float*
  // CHECK: [[ELT_VAL:%.+]] = load float, float* [[ELT]], align
  //
  // CHECK: [[ELT_CAST:%.+]] = bitcast float [[ELT_VAL]] to i32
  // CHECK: [[WS32:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[WS:%.+]] = trunc i32 [[WS32]] to i16
  // CHECK: [[REMOTE_ELT2_VAL32:%.+]] = call i32 @__kmpc_shuffle_int32(i32 [[ELT_CAST]], i16 [[LANEOFFSET]], i16 [[WS]])
  // CHECK: [[REMOTE_ELT2_VAL:%.+]] = bitcast i32 [[REMOTE_ELT2_VAL32]] to float
  //
  // CHECK: store float [[REMOTE_ELT2_VAL]], float* [[REMOTE_ELT2]], align
  // CHECK: [[REMOTE_ELT2C:%.+]] = bitcast float* [[REMOTE_ELT2]] to i8*
  // CHECK: store i8* [[REMOTE_ELT2C]], i8** [[REMOTE_ELT_REF]], align
  //
  // Condition to reduce
  // CHECK: [[CONDALG0:%.+]] = icmp eq i16 [[ALGVER]], 0
  //
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp ult i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[CONDALG1:%.+]] = and i1 [[COND1]], [[COND2]]
  //
  // CHECK: [[COND3:%.+]] = icmp eq i16 [[ALGVER]], 2
  // CHECK: [[COND4:%.+]] = and i16 [[LANEID]], 1
  // CHECK: [[COND5:%.+]] = icmp eq i16 [[COND4]], 0
  // CHECK: [[COND6:%.+]] = and i1 [[COND3]], [[COND5]]
  // CHECK: [[COND7:%.+]] = icmp sgt i16 [[LANEOFFSET]], 0
  // CHECK: [[CONDALG2:%.+]] = and i1 [[COND6]], [[COND7]]
  //
  // CHECK: [[COND8:%.+]] = or i1 [[CONDALG0]], [[CONDALG1]]
  // CHECK: [[SHOULD_REDUCE:%.+]] = or i1 [[COND8]], [[CONDALG2]]
  // CHECK: br i1 [[SHOULD_REDUCE]], label {{%?}}[[DO_REDUCE:.+]], label {{%?}}[[REDUCE_ELSE:.+]]
  //
  // CHECK: [[DO_REDUCE]]
  // CHECK: [[RED_LIST1_VOID:%.+]] = bitcast [[RLT]]* [[RED_LIST]] to i8*
  // CHECK: [[RED_LIST2_VOID:%.+]] = bitcast [[RLT]]* [[REMOTE_RED_LIST]] to i8*
  // CHECK: call void [[REDUCTION_FUNC]](i8* [[RED_LIST1_VOID]], i8* [[RED_LIST2_VOID]])
  // CHECK: br label {{%?}}[[REDUCE_CONT:.+]]
  //
  // CHECK: [[REDUCE_ELSE]]
  // CHECK: br label {{%?}}[[REDUCE_CONT]]
  //
  // CHECK: [[REDUCE_CONT]]
  // Now check if we should just copy over the remote reduction list
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp uge i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[SHOULD_COPY:%.+]] = and i1 [[COND1]], [[COND2]]
  // CHECK: br i1 [[SHOULD_COPY]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // CHECK: [[DO_COPY]]
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = load i8*, i8** [[REMOTE_ELT_REF]],
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = load i8, i8* [[REMOTE_ELT_VOID]], align
  // CHECK: store i8 [[REMOTE_ELT_VAL]], i8* [[ELT_VOID]], align
  //
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = load i8*, i8** [[REMOTE_ELT_REF]],
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT:%.+]] = bitcast i8* [[REMOTE_ELT_VOID]] to float*
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = load float, float* [[REMOTE_ELT]], align
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to float*
  // CHECK: store float [[REMOTE_ELT_VAL]], float* [[ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // CHECK: [[COPY_CONT]]
  // CHECK: void

  //
  // Inter warp copy function
  // CHECK: define internal void [[WARP_COPY_FN]](i8*, i32)
  // CHECK-DAG: [[LANEID:%.+]] = and i32 {{.+}}, 31
  // CHECK-DAG: [[WARPID:%.+]] = ashr i32 {{.+}}, 5
  // CHECK-DAG: [[RED_LIST:%.+]] = bitcast i8* {{.+}} to [[RLT]]*
  // CHECK: [[IS_WARP_MASTER:%.+]] = icmp eq i32 [[LANEID]], 0
  // CHECK: br i1 [[IS_WARP_MASTER]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // [[DO_COPY]]
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT_VAL:%.+]] = load i8, i8* [[ELT_VOID]], align
  //
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[WARPID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i8 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: store i8 [[ELT_VAL]], i8 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // Barrier after copy to shared memory storage medium.
  // CHECK: [[COPY_CONT]]
  // CHECK: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[ACTIVE_THREADS:%.+]] = mul nsw i32 [[ACTIVE_WARPS:%.+]], [[WS]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  //
  // Read into warp 0.
  // CHECK: [[IS_W0_ACTIVE_THREAD:%.+]] = icmp ult i32 [[TID:%.+]], [[ACTIVE_WARPS]]
  // CHECK: br i1 [[IS_W0_ACTIVE_THREAD]], label {{%?}}[[DO_READ:.+]], label {{%?}}[[READ_ELSE:.+]]
  //
  // CHECK: [[DO_READ]]
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[TID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i8 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: [[MEDIUM_ELT_VAL:%.+]] = load i8, i8 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: store i8 [[MEDIUM_ELT_VAL]], i8* [[ELT_VOID]], align
  // CHECK: br label {{%?}}[[READ_CONT:.+]]
  //
  // CHECK: [[READ_ELSE]]
  // CHECK: br label {{%?}}[[READ_CONT]]
  //
  // CHECK: [[READ_CONT]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  // CHECK: [[IS_WARP_MASTER:%.+]] = icmp eq i32 [[LANEID]], 0
  // CHECK: br i1 [[IS_WARP_MASTER]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // [[DO_COPY]]
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to float*
  // CHECK: [[ELT_VAL:%.+]] = load float, float* [[ELT]], align
  //
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[WARPID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to float addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: store float [[ELT_VAL]], float addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // Barrier after copy to shared memory storage medium.
  // CHECK: [[COPY_CONT]]
  // CHECK: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[ACTIVE_THREADS:%.+]] = mul nsw i32 [[ACTIVE_WARPS:%.+]], [[WS]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  //
  // Read into warp 0.
  // CHECK: [[IS_W0_ACTIVE_THREAD:%.+]] = icmp ult i32 [[TID:%.+]], [[ACTIVE_WARPS]]
  // CHECK: br i1 [[IS_W0_ACTIVE_THREAD]], label {{%?}}[[DO_READ:.+]], label {{%?}}[[READ_ELSE:.+]]
  //
  // CHECK: [[DO_READ]]
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[TID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to float addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: [[MEDIUM_ELT_VAL:%.+]] = load float, float addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to float*
  // CHECK: store float [[MEDIUM_ELT_VAL]], float* [[ELT]], align
  // CHECK: br label {{%?}}[[READ_CONT:.+]]
  //
  // CHECK: [[READ_ELSE]]
  // CHECK: br label {{%?}}[[READ_CONT]]
  //
  // CHECK: [[READ_CONT]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  // CHECK: ret










  // CHECK: define {{.*}}void {{@__omp_offloading_.+template.+l38}}(
  //
  // CHECK: call void @__kmpc_spmd_kernel_init(
  // CHECK: br label {{%?}}[[EXECUTE:.+]]
  //
  // CHECK: [[EXECUTE]]
  // CHECK: {{call|invoke}} void [[PFN2:@.+]](i32*
  // CHECK: call void @__kmpc_spmd_kernel_deinit()
  //
  //
  // define internal void [[PFN2]](
  // CHECK: store i32 0, i32* [[A:%.+]], align
  // CHECK: store i16 -32768, i16* [[B:%.+]], align
  // CHECK: [[A_VAL:%.+]] = load i32, i32* [[A:%.+]], align
  // CHECK: [[OR:%.+]] = or i32 [[A_VAL]], 1
  // CHECK: store i32 [[OR]], i32* [[A]], align
  // CHECK: [[BV16:%.+]] = load i16, i16* [[B]], align
  // CHECK: [[BV:%.+]] = sext i16 [[BV16]] to i32
  // CHECK: [[CMP:%.+]] = icmp sgt i32 99, [[BV]]
  // CHECK: br i1 [[CMP]], label {{%?}}[[DO_MAX:.+]], label {{%?}}[[MAX_ELSE:.+]]
  //
  // CHECK: [[DO_MAX]]
  // CHECK: br label {{%?}}[[MAX_CONT:.+]]
  //
  // CHECK: [[MAX_ELSE]]
  // CHECK: [[BV:%.+]] = load i16, i16* [[B]], align
  // CHECK: [[MAX:%.+]] = sext i16 [[BV]] to i32
  // CHECK: br label {{%?}}[[MAX_CONT]]
  //
  // CHECK: [[MAX_CONT]]
  // CHECK: [[B_LVALUE:%.+]] = phi i32 [ 99, %[[DO_MAX]] ], [ [[MAX]], %[[MAX_ELSE]] ]
  // CHECK: [[TRUNC:%.+]] = trunc i32 [[B_LVALUE]] to i16
  // CHECK: store i16 [[TRUNC]], i16* [[B]], align
  // CHECK: [[PTR1:%.+]] = getelementptr inbounds [[RLT:.+]], [2 x i8*]* [[RL:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[A_CAST:%.+]] = bitcast i32* [[A]] to i8*
  // CHECK: store i8* [[A_CAST]], i8** [[PTR1]], align
  // CHECK: [[PTR2:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RL]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[B_CAST:%.+]] = bitcast i16* [[B]] to i8*
  // CHECK: store i8* [[B_CAST]], i8** [[PTR2]], align
  // CHECK: [[ARG_RL:%.+]] = bitcast [[RLT]]* [[RL]] to i8*
  // CHECK: [[RET:%.+]] = call i32 @__kmpc_nvptx_parallel_reduce_nowait(i32 {{.+}}, i32 2, i{{32|64}} {{8|16}}, i8* [[ARG_RL]], void (i8*, i16, i16, i16)* [[SHUFFLE_REDUCE_FN:@.+]], void (i8*, i32)* [[WARP_COPY_FN:@.+]])
  // CHECK: switch i32 [[RET]], label {{%?}}[[DEFAULTLABEL:.+]] [
  // CHECK: i32 1, label {{%?}}[[REDLABEL:.+]]

  // CHECK: [[REDLABEL]]
  // CHECK: [[A_INV:%.+]] = load i32, i32* [[A_IN:%.+]], align
  // CHECK: [[AV:%.+]] = load i32, i32* [[A]], align
  // CHECK: [[OR:%.+]] = or i32 [[A_INV]], [[AV]]
  // CHECK: store i32 [[OR]], i32* [[A_IN]], align
  // CHECK: [[B_INV16:%.+]] = load i16, i16* [[B_IN:%.+]], align
  // CHECK: [[B_INV:%.+]] = sext i16 [[B_INV16]] to i32
  // CHECK: [[BV16:%.+]] = load i16, i16* [[B]], align
  // CHECK: [[BV:%.+]] = sext i16 [[BV16]] to i32
  // CHECK: [[CMP:%.+]] = icmp sgt i32 [[B_INV]], [[BV]]
  // CHECK: br i1 [[CMP]], label {{%?}}[[DO_MAX:.+]], label {{%?}}[[MAX_ELSE:.+]]
  //
  // CHECK: [[DO_MAX]]
  // CHECK: [[MAX1:%.+]] = load i16, i16* [[B_IN]], align
  // CHECK: br label {{%?}}[[MAX_CONT:.+]]
  //
  // CHECK: [[MAX_ELSE]]
  // CHECK: [[MAX2:%.+]] = load i16, i16* [[B]], align
  // CHECK: br label {{%?}}[[MAX_CONT]]
  //
  // CHECK: [[MAX_CONT]]
  // CHECK: [[B_MAX:%.+]] = phi i16 [ [[MAX1]], %[[DO_MAX]] ], [ [[MAX2]], %[[MAX_ELSE]] ]
  // CHECK: store i16 [[B_MAX]], i16* [[B_IN]], align
  // CHECK: call void @__kmpc_nvptx_end_reduce_nowait(
  // CHECK: br label %[[DEFAULTLABEL]]
  //
  // CHECK: [[DEFAULTLABEL]]
  // CHECK: ret

  //
  // Reduction function
  // CHECK: define internal void [[REDUCTION_FUNC:@.+]](i8*, i8*)
  // CHECK: [[VAR1_RHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_RHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR1_RHS_VOID:%.+]] = load i8*, i8** [[VAR1_RHS_REF]],
  // CHECK: [[VAR1_RHS:%.+]] = bitcast i8* [[VAR1_RHS_VOID]] to i32*
  //
  // CHECK: [[VAR1_LHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_LHS:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[VAR1_LHS_VOID:%.+]] = load i8*, i8** [[VAR1_LHS_REF]],
  // CHECK: [[VAR1_LHS:%.+]] = bitcast i8* [[VAR1_LHS_VOID]] to i32*
  //
  // CHECK: [[VAR2_RHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_RHS]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[VAR2_RHS_VOID:%.+]] = load i8*, i8** [[VAR2_RHS_REF]],
  // CHECK: [[VAR2_RHS:%.+]] = bitcast i8* [[VAR2_RHS_VOID]] to i16*
  //
  // CHECK: [[VAR2_LHS_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST_LHS]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[VAR2_LHS_VOID:%.+]] = load i8*, i8** [[VAR2_LHS_REF]],
  // CHECK: [[VAR2_LHS:%.+]] = bitcast i8* [[VAR2_LHS_VOID]] to i16*
  //
  // CHECK: [[VAR1_LHS_VAL:%.+]] = load i32, i32* [[VAR1_LHS]],
  // CHECK: [[VAR1_RHS_VAL:%.+]] = load i32, i32* [[VAR1_RHS]],
  // CHECK: [[OR:%.+]] = or i32 [[VAR1_LHS_VAL]], [[VAR1_RHS_VAL]]
  // CHECK: store i32 [[OR]], i32* [[VAR1_LHS]],
  //
  // CHECK: [[VAR2_LHS_VAL16:%.+]] = load i16, i16* [[VAR2_LHS]],
  // CHECK: [[VAR2_LHS_VAL:%.+]] = sext i16 [[VAR2_LHS_VAL16]] to i32
  // CHECK: [[VAR2_RHS_VAL16:%.+]] = load i16, i16* [[VAR2_RHS]],
  // CHECK: [[VAR2_RHS_VAL:%.+]] = sext i16 [[VAR2_RHS_VAL16]] to i32
  //
  // CHECK: [[CMP:%.+]] = icmp sgt i32 [[VAR2_LHS_VAL]], [[VAR2_RHS_VAL]]
  // CHECK: br i1 [[CMP]], label {{%?}}[[DO_MAX:.+]], label {{%?}}[[MAX_ELSE:.+]]
  //
  // CHECK: [[DO_MAX]]
  // CHECK: [[MAX1:%.+]] = load i16, i16* [[VAR2_LHS]], align
  // CHECK: br label {{%?}}[[MAX_CONT:.+]]
  //
  // CHECK: [[MAX_ELSE]]
  // CHECK: [[MAX2:%.+]] = load i16, i16* [[VAR2_RHS]], align
  // CHECK: br label {{%?}}[[MAX_CONT]]
  //
  // CHECK: [[MAX_CONT]]
  // CHECK: [[MAXV:%.+]] = phi i16 [ [[MAX1]], %[[DO_MAX]] ], [ [[MAX2]], %[[MAX_ELSE]] ]
  // CHECK: store i16 [[MAXV]], i16* [[VAR2_LHS]],
  // CHECK: ret void

  //
  // Shuffle and reduce function
  // CHECK: define internal void [[SHUFFLE_REDUCE_FN]](i8*, i16 {{.*}}, i16 {{.*}}, i16 {{.*}})
  // CHECK: [[REMOTE_RED_LIST:%.+]] = alloca [[RLT]], align
  // CHECK: [[REMOTE_ELT1:%.+]] = alloca i32
  // CHECK: [[REMOTE_ELT2:%.+]] = alloca i16
  //
  // CHECK: [[LANEID:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[LANEOFFSET:%.+]] = load i16, i16* {{.+}}, align
  // CHECK: [[ALGVER:%.+]] = load i16, i16* {{.+}}, align
  //
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i32*
  // CHECK: [[ELT_VAL:%.+]] = load i32, i32* [[ELT]], align
  //
  // CHECK: [[WS32:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[WS:%.+]] = trunc i32 [[WS32]] to i16
  // CHECK: [[REMOTE_ELT1_VAL:%.+]] = call i32 @__kmpc_shuffle_int32(i32 [[ELT_VAL]], i16 [[LANEOFFSET]], i16 [[WS]])
  //
  // CHECK: store i32 [[REMOTE_ELT1_VAL]], i32* [[REMOTE_ELT1]], align
  // CHECK: [[REMOTE_ELT1C:%.+]] = bitcast i32* [[REMOTE_ELT1]] to i8*
  // CHECK: store i8* [[REMOTE_ELT1C]], i8** [[REMOTE_ELT_REF]], align
  //
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i16*
  // CHECK: [[ELT_VAL:%.+]] = load i16, i16* [[ELT]], align
  //
  // CHECK: [[ELT_CAST:%.+]] = sext i16 [[ELT_VAL]] to i32
  // CHECK: [[WS32:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[WS:%.+]] = trunc i32 [[WS32]] to i16
  // CHECK: [[REMOTE_ELT2_VAL32:%.+]] = call i32 @__kmpc_shuffle_int32(i32 [[ELT_CAST]], i16 [[LANEOFFSET]], i16 [[WS]])
  // CHECK: [[REMOTE_ELT2_VAL:%.+]] = trunc i32 [[REMOTE_ELT2_VAL32]] to i16
  //
  // CHECK: store i16 [[REMOTE_ELT2_VAL]], i16* [[REMOTE_ELT2]], align
  // CHECK: [[REMOTE_ELT2C:%.+]] = bitcast i16* [[REMOTE_ELT2]] to i8*
  // CHECK: store i8* [[REMOTE_ELT2C]], i8** [[REMOTE_ELT_REF]], align
  //
  // Condition to reduce
  // CHECK: [[CONDALG0:%.+]] = icmp eq i16 [[ALGVER]], 0
  //
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp ult i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[CONDALG1:%.+]] = and i1 [[COND1]], [[COND2]]
  //
  // CHECK: [[COND3:%.+]] = icmp eq i16 [[ALGVER]], 2
  // CHECK: [[COND4:%.+]] = and i16 [[LANEID]], 1
  // CHECK: [[COND5:%.+]] = icmp eq i16 [[COND4]], 0
  // CHECK: [[COND6:%.+]] = and i1 [[COND3]], [[COND5]]
  // CHECK: [[COND7:%.+]] = icmp sgt i16 [[LANEOFFSET]], 0
  // CHECK: [[CONDALG2:%.+]] = and i1 [[COND6]], [[COND7]]
  //
  // CHECK: [[COND8:%.+]] = or i1 [[CONDALG0]], [[CONDALG1]]
  // CHECK: [[SHOULD_REDUCE:%.+]] = or i1 [[COND8]], [[CONDALG2]]
  // CHECK: br i1 [[SHOULD_REDUCE]], label {{%?}}[[DO_REDUCE:.+]], label {{%?}}[[REDUCE_ELSE:.+]]
  //
  // CHECK: [[DO_REDUCE]]
  // CHECK: [[RED_LIST1_VOID:%.+]] = bitcast [[RLT]]* [[RED_LIST]] to i8*
  // CHECK: [[RED_LIST2_VOID:%.+]] = bitcast [[RLT]]* [[REMOTE_RED_LIST]] to i8*
  // CHECK: call void [[REDUCTION_FUNC]](i8* [[RED_LIST1_VOID]], i8* [[RED_LIST2_VOID]])
  // CHECK: br label {{%?}}[[REDUCE_CONT:.+]]
  //
  // CHECK: [[REDUCE_ELSE]]
  // CHECK: br label {{%?}}[[REDUCE_CONT]]
  //
  // CHECK: [[REDUCE_CONT]]
  // Now check if we should just copy over the remote reduction list
  // CHECK: [[COND1:%.+]] = icmp eq i16 [[ALGVER]], 1
  // CHECK: [[COND2:%.+]] = icmp uge i16 [[LANEID]], [[LANEOFFSET]]
  // CHECK: [[SHOULD_COPY:%.+]] = and i1 [[COND1]], [[COND2]]
  // CHECK: br i1 [[SHOULD_COPY]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // CHECK: [[DO_COPY]]
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = load i8*, i8** [[REMOTE_ELT_REF]],
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT:%.+]] = bitcast i8* [[REMOTE_ELT_VOID]] to i32*
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = load i32, i32* [[REMOTE_ELT]], align
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i32*
  // CHECK: store i32 [[REMOTE_ELT_VAL]], i32* [[ELT]], align
  //
  // CHECK: [[REMOTE_ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[REMOTE_RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[REMOTE_ELT_VOID:%.+]] = load i8*, i8** [[REMOTE_ELT_REF]],
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[REMOTE_ELT:%.+]] = bitcast i8* [[REMOTE_ELT_VOID]] to i16*
  // CHECK: [[REMOTE_ELT_VAL:%.+]] = load i16, i16* [[REMOTE_ELT]], align
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i16*
  // CHECK: store i16 [[REMOTE_ELT_VAL]], i16* [[ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // CHECK: [[COPY_CONT]]
  // CHECK: void

  //
  // Inter warp copy function
  // CHECK: define internal void [[WARP_COPY_FN]](i8*, i32)
  // CHECK-DAG: [[LANEID:%.+]] = and i32 {{.+}}, 31
  // CHECK-DAG: [[WARPID:%.+]] = ashr i32 {{.+}}, 5
  // CHECK-DAG: [[RED_LIST:%.+]] = bitcast i8* {{.+}} to [[RLT]]*
  // CHECK: [[IS_WARP_MASTER:%.+]] = icmp eq i32 [[LANEID]], 0
  // CHECK: br i1 [[IS_WARP_MASTER]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // [[DO_COPY]]
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i32*
  // CHECK: [[ELT_VAL:%.+]] = load i32, i32* [[ELT]], align
  //
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[WARPID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i32 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: store i32 [[ELT_VAL]], i32 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // Barrier after copy to shared memory storage medium.
  // CHECK: [[COPY_CONT]]
  // CHECK: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[ACTIVE_THREADS:%.+]] = mul nsw i32 [[ACTIVE_WARPS:%.+]], [[WS]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  //
  // Read into warp 0.
  // CHECK: [[IS_W0_ACTIVE_THREAD:%.+]] = icmp ult i32 [[TID:%.+]], [[ACTIVE_WARPS]]
  // CHECK: br i1 [[IS_W0_ACTIVE_THREAD]], label {{%?}}[[DO_READ:.+]], label {{%?}}[[READ_ELSE:.+]]
  //
  // CHECK: [[DO_READ]]
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[TID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i32 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: [[MEDIUM_ELT_VAL:%.+]] = load i32, i32 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i32*
  // CHECK: store i32 [[MEDIUM_ELT_VAL]], i32* [[ELT]], align
  // CHECK: br label {{%?}}[[READ_CONT:.+]]
  //
  // CHECK: [[READ_ELSE]]
  // CHECK: br label {{%?}}[[READ_CONT]]
  //
  // CHECK: [[READ_CONT]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  // CHECK: [[IS_WARP_MASTER:%.+]] = icmp eq i32 [[LANEID]], 0
  // CHECK: br i1 [[IS_WARP_MASTER]], label {{%?}}[[DO_COPY:.+]], label {{%?}}[[COPY_ELSE:.+]]
  //
  // [[DO_COPY]]
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i16*
  // CHECK: [[ELT_VAL:%.+]] = load i16, i16* [[ELT]], align
  //
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[WARPID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i16 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: store i16 [[ELT_VAL]], i16 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: br label {{%?}}[[COPY_CONT:.+]]
  //
  // CHECK: [[COPY_ELSE]]
  // CHECK: br label {{%?}}[[COPY_CONT]]
  //
  // Barrier after copy to shared memory storage medium.
  // CHECK: [[COPY_CONT]]
  // CHECK: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK: [[ACTIVE_THREADS:%.+]] = mul nsw i32 [[ACTIVE_WARPS:%.+]], [[WS]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  //
  // Read into warp 0.
  // CHECK: [[IS_W0_ACTIVE_THREAD:%.+]] = icmp ult i32 [[TID:%.+]], [[ACTIVE_WARPS]]
  // CHECK: br i1 [[IS_W0_ACTIVE_THREAD]], label {{%?}}[[DO_READ:.+]], label {{%?}}[[READ_ELSE:.+]]
  //
  // CHECK: [[DO_READ]]
  // CHECK: [[MEDIUM_ELT64:%.+]] = getelementptr inbounds [32 x i64], [32 x i64] addrspace([[SHARED_ADDRSPACE]])* [[TRANSFER_STORAGE]], i64 0, i32 [[TID]]
  // CHECK: [[MEDIUM_ELT:%.+]] = bitcast i64 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT64]] to i16 addrspace([[SHARED_ADDRSPACE]])*
  // CHECK: [[MEDIUM_ELT_VAL:%.+]] = load i16, i16 addrspace([[SHARED_ADDRSPACE]])* [[MEDIUM_ELT]], align
  // CHECK: [[ELT_REF:%.+]] = getelementptr inbounds [[RLT]], [[RLT]]* [[RED_LIST:%.+]], i{{32|64}} 0, i{{32|64}} 1
  // CHECK: [[ELT_VOID:%.+]] = load i8*, i8** [[ELT_REF]],
  // CHECK: [[ELT:%.+]] = bitcast i8* [[ELT_VOID]] to i16*
  // CHECK: store i16 [[MEDIUM_ELT_VAL]], i16* [[ELT]], align
  // CHECK: br label {{%?}}[[READ_CONT:.+]]
  //
  // CHECK: [[READ_ELSE]]
  // CHECK: br label {{%?}}[[READ_CONT]]
  //
  // CHECK: [[READ_CONT]]
  // CHECK: call void @llvm.nvvm.barrier(i32 1, i32 [[ACTIVE_THREADS]])
  // CHECK: ret

#endif
