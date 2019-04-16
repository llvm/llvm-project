// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -verify -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// Check that the execution mode of all 7 target regions is set to Generic Mode.
// CHECK-DAG: [[NONSPMD:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds
// CHECK-DAG: [[UNKNOWN:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 2, i32 0, i8* getelementptr inbounds
// CHECK-DAG: {{@__omp_offloading_.+l45}}_exec_mode = weak constant i8 0
// CHECK-DAG: {{@__omp_offloading_.+l123}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l200}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l310}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l348}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l366}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l331}}_exec_mode = weak constant i8 1

__thread int id;

int baz(int f, double &a);

template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
  tx &operator[](int i) { return X; }
};

// CHECK: define weak void @__omp_offloading_{{.+}}_{{.+}}targetBar{{.+}}_l45(i32* [[PTR1:%.+]], i32** dereferenceable{{.*}} [[PTR2_REF:%.+]])
// CHECK: store i32* [[PTR1]], i32** [[PTR1_ADDR:%.+]],
// CHECK: store i32** [[PTR2_REF]], i32*** [[PTR2_REF_PTR:%.+]],
// CHECK: [[PTR2_REF:%.+]] = load i32**, i32*** [[PTR2_REF_PTR]],
// CHECK: call void @__kmpc_spmd_kernel_init(
// CHECK: call void @__kmpc_data_sharing_init_stack_spmd()
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* @{{.+}})
// CHECK: store i32 [[GTID]], i32* [[THREADID:%.+]],
// CHECK: call void @{{.+}}(i32* [[THREADID]], i32* %{{.+}}, i32** [[PTR1_ADDR]], i32** [[PTR2_REF]])
// CHECK: call void @__kmpc_spmd_kernel_deinit_v2(i16 1)
void targetBar(int *Ptr1, int *Ptr2) {
#pragma omp target map(Ptr1[:0], Ptr2)
#pragma omp parallel num_threads(2)
  *Ptr1 = *Ptr2;
}

int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+foo.+l123}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T1:@__omp_offloading_.+foo.+l123]]()
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
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void
  #pragma omp target
  {
  }

  // CHECK-NOT: define {{.*}}void [[T2:@__omp_offloading_.+foo.+]]_worker()
  #pragma omp target if(0)
  {
  }

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+foo.+l200}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T2:@__omp_offloading_.+foo.+l200]](i[[SZ:32|64]] [[ARG1:%[a-zA-Z_]+]], i[[SZ:32|64]] [[ID:%[a-zA-Z_]+]])
  // CHECK: [[AA_ADDR:%.+]] = alloca i[[SZ]],
  // CHECK: store i[[SZ]] [[ARG1]], i[[SZ]]* [[AA_ADDR]],
  // CHECK: [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
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
  // CHECK: load i16, i16* [[AA_CADDR]],
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void
  #pragma omp target if(1)
  {
    aa += 1;
    id = aa;
  }

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+foo.+l310}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T3:@__omp_offloading_.+foo.+l310]](i[[SZ]]
  // Create local storage for each capture.
  // CHECK:    [[LOCAL_A:%.+]] = alloca i[[SZ]]
  // CHECK:    [[LOCAL_B:%.+]] = alloca [10 x float]*
  // CHECK:    [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
  // CHECK:    [[LOCAL_BN:%.+]] = alloca float*
  // CHECK:    [[LOCAL_C:%.+]] = alloca [5 x [10 x double]]*
  // CHECK:    [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
  // CHECK:    [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]
  // CHECK:    [[LOCAL_CN:%.+]] = alloca double*
  // CHECK:    [[LOCAL_D:%.+]] = alloca [[TT:%.+]]*
  // CHECK-DAG: store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
  // CHECK-DAG: store [10 x float]* [[ARG_B:%.+]], [10 x float]** [[LOCAL_B]]
  // CHECK-DAG: store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
  // CHECK-DAG: store float* [[ARG_BN:%.+]], float** [[LOCAL_BN]]
  // CHECK-DAG: store [5 x [10 x double]]* [[ARG_C:%.+]], [5 x [10 x double]]** [[LOCAL_C]]
  // CHECK-DAG: store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
  // CHECK-DAG: store i[[SZ]] [[ARG_VLA3:%.+]], i[[SZ]]* [[LOCAL_VLA3]]
  // CHECK-DAG: store double* [[ARG_CN:%.+]], double** [[LOCAL_CN]]
  // CHECK-DAG: store [[TT]]* [[ARG_D:%.+]], [[TT]]** [[LOCAL_D]]
  //
  // CHECK-64-DAG: [[REF_A:%.+]] = bitcast i64* [[LOCAL_A]] to i32*
  // CHECK-DAG:    [[REF_B:%.+]] = load [10 x float]*, [10 x float]** [[LOCAL_B]],
  // CHECK-DAG:    [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
  // CHECK-DAG:    [[REF_BN:%.+]] = load float*, float** [[LOCAL_BN]],
  // CHECK-DAG:    [[REF_C:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[LOCAL_C]],
  // CHECK-DAG:    [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
  // CHECK-DAG:    [[VAL_VLA3:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA3]],
  // CHECK-DAG:    [[REF_CN:%.+]] = load double*, double** [[LOCAL_CN]],
  // CHECK-DAG:    [[REF_D:%.+]] = load [[TT]]*, [[TT]]** [[LOCAL_D]],
  //
  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void [[T3]]_worker()
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
  // Use captures.
  // CHECK-64-DAG:  load i32, i32* [[REF_A]]
  // CHECK-32-DAG:  load i32, i32* [[LOCAL_A]]
  // CHECK-DAG:  getelementptr inbounds [10 x float], [10 x float]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
  // CHECK-DAG:  getelementptr inbounds float, float* [[REF_BN]], i[[SZ]] 3
  // CHECK-DAG:  getelementptr inbounds [5 x [10 x double]], [5 x [10 x double]]* [[REF_C]], i[[SZ]] 0, i[[SZ]] 1
  // CHECK-DAG:  getelementptr inbounds double, double* [[REF_CN]], i[[SZ]] %{{.+}}
  // CHECK-DAG:     getelementptr inbounds [[TT]], [[TT]]* [[REF_D]], i32 0, i32 0
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
  #pragma omp target if(n>20)
  {
    a += 1;
    b[2] += 1.0;
    bn[3] += 1.0;
    c[1][2] += 1.0;
    cn[1][3] += 1.0;
    d.X += 1;
    d.Y += 1;
    d[0] += 1;
  }

  return a;
}

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target if(n>40)
  {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

  #pragma omp target if(n>50)
  {
    a += 1;
    aa += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

    #pragma omp target if(n>60)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
      baz(a, a);
    }

    return c[1][1] + (int)b;
  }
};

int bar(int n){
  int a = 0;

  a += foo(n);

  S1 S;
  a += S.r1(n);

  a += fstatic(n);

  a += ftemplate<int>(n);

  return a;
}

int baz(int f, double &a) {
#pragma omp parallel
  f = 2 + a;
  return f;
}

  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+static.+348}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T4:@__omp_offloading_.+static.+l348]](i[[SZ]]
  // Create local storage for each capture.
  // CHECK:  [[LOCAL_A:%.+]] = alloca i[[SZ]]
  // CHECK:  [[LOCAL_AA:%.+]] = alloca i[[SZ]]
  // CHECK:  [[LOCAL_AAA:%.+]] = alloca i[[SZ]]
  // CHECK:  [[LOCAL_B:%.+]] = alloca [10 x i32]*
  // CHECK-DAG:  store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
  // CHECK-DAG:  store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
  // CHECK-DAG:  store i[[SZ]] [[ARG_AAA:%.+]], i[[SZ]]* [[LOCAL_AAA]]
  // CHECK-DAG:  store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
  // Store captures in the context.
  // CHECK-64-DAG:   [[REF_A:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
  // CHECK-DAG:      [[REF_AA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
  // CHECK-DAG:      [[REF_AAA:%.+]] = bitcast i[[SZ]]* [[LOCAL_AAA]] to i8*
  // CHECK-DAG:      [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],
  //
  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void [[T4]]_worker()
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
  // CHECK-64-DAG: load i32, i32* [[REF_A]]
  // CHECK-32-DAG: load i32, i32* [[LOCAL_A]]
  // CHECK-DAG:    load i16, i16* [[REF_AA]]
  // CHECK-DAG:    getelementptr inbounds [10 x i32], [10 x i32]* [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void



  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+S1.+l366}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[NONSPMD]]
  // CHECK: [[WORK_FN:%.+]] = bitcast i8* [[WORK]] to void (i16, i32)*
  // CHECK: call void [[WORK_FN]](i16 0, i32 [[GTID]])
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T5:@__omp_offloading_.+S1.+l366]](
  // Create local storage for each capture.
  // CHECK:       [[LOCAL_THIS:%.+]] = alloca [[S1:%struct.*]]*
  // CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
  // CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
  // CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
  // CHECK:       [[LOCAL_C:%.+]] = alloca i16*
  // CHECK-DAG:   store [[S1]]* [[ARG_THIS:%.+]], [[S1]]** [[LOCAL_THIS]]
  // CHECK-DAG:   store i[[SZ]] [[ARG_B:%.+]], i[[SZ]]* [[LOCAL_B]]
  // CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
  // CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
  // CHECK-DAG:   store i16* [[ARG_C:%.+]], i16** [[LOCAL_C]]
  // Store captures in the context.
  // CHECK-DAG:   [[REF_THIS:%.+]] = load [[S1]]*, [[S1]]** [[LOCAL_THIS]],
  // CHECK-64-DAG:[[REF_B:%.+]] = bitcast i[[SZ]]* [[LOCAL_B]] to i32*
  // CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
  // CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
  // CHECK-DAG:   [[REF_C:%.+]] = load i16*, i16** [[LOCAL_C]],
  //
  // CHECK-DAG: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-DAG: [[NTH:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  // CHECK-DAG: [[WS:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.warpsize()
  // CHECK-DAG: [[TH_LIMIT:%.+]] = sub nuw i32 [[NTH]], [[WS]]
  // CHECK: [[IS_WORKER:%.+]] = icmp ult i32 [[TID]], [[TH_LIMIT]]
  // CHECK: br i1 [[IS_WORKER]], label {{%?}}[[WORKER:.+]], label {{%?}}[[CHECK_MASTER:.+]]
  //
  // CHECK: [[WORKER]]
  // CHECK: {{call|invoke}} void [[T5]]_worker()
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
  // Use captures.
  // CHECK-DAG:   getelementptr inbounds [[S1]], [[S1]]* [[REF_THIS]], i32 0, i32 0
  // CHECK-64-DAG:load i32, i32* [[REF_B]]
  // CHECK-32-DAG:load i32, i32* [[LOCAL_B]]
  // CHECK-DAG:   getelementptr inbounds i16, i16* [[REF_C]], i[[SZ]] %{{.+}}
  // CHECK: call i32 [[BAZ:@.*baz.*]](i32 %
  // CHECK: br label {{%?}}[[TERMINATE:.+]]
  //
  // CHECK: [[TERMINATE]]
  // CHECK: call void @__kmpc_kernel_deinit(
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[EXIT]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define i32 [[BAZ]](i32 [[F:%.*]], double* dereferenceable{{.*}})
  // CHECK: alloca i32,
  // CHECK: [[LOCAL_F_PTR:%.+]] = alloca i32,
  // CHECK: [[ZERO_ADDR:%.+]] = alloca i32,
  // CHECK: store i32 0, i32* [[ZERO_ADDR]]
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[UNKNOWN]]
  // CHECK: [[PAR_LEVEL:%.+]] = call i16 @__kmpc_parallel_level(%struct.ident_t* [[UNKNOWN]], i32 [[GTID]])
  // CHECK: [[IS_TTD:%.+]] = icmp eq i16 %1, 0
  // CHECK: [[RES:%.+]] = call i8 @__kmpc_is_spmd_exec_mode()
  // CHECK: [[IS_SPMD:%.+]] = icmp ne i8 [[RES]], 0
  // CHECK: br i1 [[IS_SPMD]], label
  // CHECK: br label
  // CHECK: [[SIZE:%.+]] = select i1 [[IS_TTD]], i{{64|32}} 4, i{{64|32}} 128
  // CHECK: [[PTR:%.+]] = call i8* @__kmpc_data_sharing_coalesced_push_stack(i{{64|32}} [[SIZE]], i16 0)
  // CHECK: [[REC_ADDR:%.+]] = bitcast i8* [[PTR]] to [[GLOBAL_ST:%.+]]*
  // CHECK: br label
  // CHECK: [[ITEMS:%.+]] = phi [[GLOBAL_ST]]* [ null, {{.+}} ], [ [[REC_ADDR]], {{.+}} ]
  // CHECK: [[TTD_ITEMS:%.+]] = bitcast [[GLOBAL_ST]]* [[ITEMS]] to [[SEC_GLOBAL_ST:%.+]]*
  // CHECK: [[F_PTR_ARR:%.+]] = getelementptr inbounds [[GLOBAL_ST]], [[GLOBAL_ST]]* [[ITEMS]], i32 0, i32 0
  // CHECK: [[TID:%.+]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LID:%.+]] = and i32 [[TID]], 31
  // CHECK: [[GLOBAL_F_PTR_PAR:%.+]] = getelementptr inbounds [32 x i32], [32 x i32]* [[F_PTR_ARR]], i32 0, i32 [[LID]]
  // CHECK: [[GLOBAL_F_PTR_TTD:%.+]] = getelementptr inbounds [[SEC_GLOBAL_ST]], [[SEC_GLOBAL_ST]]* [[TTD_ITEMS]], i32 0, i32 0
  // CHECK: [[GLOBAL_F_PTR:%.+]] = select i1 [[IS_TTD]], i32* [[GLOBAL_F_PTR_TTD]], i32* [[GLOBAL_F_PTR_PAR]]
  // CHECK: [[F_PTR:%.+]] = select i1 [[IS_SPMD]], i32* [[LOCAL_F_PTR]], i32* [[GLOBAL_F_PTR]]
  // CHECK: store i32 %{{.+}}, i32* [[F_PTR]],

  // CHECK: [[RES:%.+]] = call i8 @__kmpc_is_spmd_exec_mode()
  // CHECK: icmp ne i8 [[RES]], 0
  // CHECK: br i1

  // CHECK: [[RES:%.+]] = call i16 @__kmpc_parallel_level(%struct.ident_t* [[UNKNOWN]], i32 [[GTID]])
  // CHECK: icmp ne i16 [[RES]], 0
  // CHECK: br i1

  // CHECK: call void @__kmpc_serialized_parallel(%struct.ident_t* [[UNKNOWN]], i32 [[GTID]])
  // CHECK: call void [[OUTLINED:@.+]](i32* [[ZERO_ADDR]], i32* [[ZERO_ADDR]], i32* [[F_PTR]], double* %{{.+}})
  // CHECK: call void @__kmpc_end_serialized_parallel(%struct.ident_t* [[UNKNOWN]], i32 [[GTID]])
  // CHECK: br label

  // CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @{{.+}} to i8*), i16 1)
  // CHECK: call void @__kmpc_begin_sharing_variables(i8*** [[SHARED_PTR:%.+]], i{{64|32}} 2)
  // CHECK: [[SHARED:%.+]] = load i8**, i8*** [[SHARED_PTR]],
  // CHECK: [[REF:%.+]] = getelementptr inbounds i8*, i8** [[SHARED]], i{{64|32}} 0
  // CHECK: [[F_REF:%.+]] = bitcast i32* [[F_PTR]] to i8*
  // CHECK: store i8* [[F_REF]], i8** [[REF]],
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: call void @__kmpc_end_sharing_variables()
  // CHECK: br label

  // CHECK: [[RES:%.+]] = load i32, i32* [[F_PTR]],
  // CHECK: store i32 [[RES]], i32* [[RET:%.+]],
  // CHECK: br i1 [[IS_SPMD]], label
  // CHECK: [[BC:%.+]] = bitcast [[GLOBAL_ST]]* [[ITEMS]] to i8*
  // CHECK: call void @__kmpc_data_sharing_pop_stack(i8* [[BC]])
  // CHECK: br label
  // CHECK: [[RES:%.+]] = load i32, i32* [[RET]],
  // CHECK: ret i32 [[RES]]


  // CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+template.+l331}}_worker()
  // CHECK-DAG: [[OMP_EXEC_STATUS:%.+]] = alloca i8,
  // CHECK-DAG: [[OMP_WORK_FN:%.+]] = alloca i8*,
  // CHECK: store i8* null, i8** [[OMP_WORK_FN]],
  // CHECK: store i8 0, i8* [[OMP_EXEC_STATUS]],
  // CHECK: br label {{%?}}[[AWAIT_WORK:.+]]
  //
  // CHECK: [[AWAIT_WORK]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: [[WORK:%.+]] = load i8*, i8** [[OMP_WORK_FN]],
  // CHECK: [[SHOULD_EXIT:%.+]] = icmp eq i8* [[WORK]], null
  // CHECK: br i1 [[SHOULD_EXIT]], label {{%?}}[[EXIT:.+]], label {{%?}}[[SEL_WORKERS:.+]]
  //
  // CHECK: [[SEL_WORKERS]]
  // CHECK: [[ST:%.+]] = load i8, i8* [[OMP_EXEC_STATUS]],
  // CHECK: [[IS_ACTIVE:%.+]] = icmp ne i8 [[ST]], 0
  // CHECK: br i1 [[IS_ACTIVE]], label {{%?}}[[EXEC_PARALLEL:.+]], label {{%?}}[[BAR_PARALLEL:.+]]
  //
  // CHECK: [[EXEC_PARALLEL]]
  // CHECK: br label {{%?}}[[TERM_PARALLEL:.+]]
  //
  // CHECK: [[TERM_PARALLEL]]
  // CHECK: br label {{%?}}[[BAR_PARALLEL]]
  //
  // CHECK: [[BAR_PARALLEL]]
  // CHECK: call void @__kmpc_barrier_simple_spmd(%struct.ident_t* null, i32 0)
  // CHECK: br label {{%?}}[[AWAIT_WORK]]
  //
  // CHECK: [[EXIT]]
  // CHECK: ret void

  // CHECK: define {{.*}}void [[T6:@__omp_offloading_.+template.+l331]](i[[SZ]]
  // Create local storage for each capture.
  // CHECK:  [[LOCAL_A:%.+]] = alloca i[[SZ]]
  // CHECK:  [[LOCAL_AA:%.+]] = alloca i[[SZ]]
  // CHECK:  [[LOCAL_B:%.+]] = alloca [10 x i32]*
  // CHECK-DAG:  store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
  // CHECK-DAG:  store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
  // CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
  // Store captures in the context.
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
  //
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

#endif
