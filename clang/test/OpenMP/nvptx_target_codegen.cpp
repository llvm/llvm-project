// XFAIL: *
// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -fopenmp -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix=CHECK2
// RUN: %clang_cc1 -no-enable-noundef-analysis -verify -Wno-vla -fopenmp -fexceptions -fcxx-exceptions -x c++ -triple nvptx-unknown-unknown -fopenmp-targets=nvptx-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - -disable-llvm-optzns | FileCheck %s --check-prefix=CHECK2
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
// CHECK-DAG: {{@__omp_offloading_.+l347}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l365}}_exec_mode = weak constant i8 1
// CHECK-DAG: {{@__omp_offloading_.+l331}}_exec_mode = weak constant i8 1

__thread int id;

int baz(int f, double &a);

template <typename tx, typename ty>
struct TT {
  tx X;
  ty Y;
  tx &operator[](int i) { return X; }
};

// CHECK: define weak void @__omp_offloading_{{.+}}_{{.+}}targetBar{{.+}}_l45(i32* [[PTR1:%.+]], i32** nonnull align {{[0-9]+}} dereferenceable{{.*}} [[PTR2_REF:%.+]])
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
#pragma omp target if (0)
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

// CHECK: define {{.*}}void [[T2:@__omp_offloading_.+foo.+l200]](i[[SZ:32|64]] [[ARG1:%[a-zA-Z_]+]])
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
#pragma omp target if (1)
  {
    aa += 1;
    aa += 2;
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
#pragma omp target if (n > 20)
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

template <typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

#pragma omp target if (n > 40)
  {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

static int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

#pragma omp target if (n > 50)
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

  int r1(int n) {
    int b = n + 1;
    short int c[2][n];

#pragma omp target if (n > 60)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
      baz(a, a);
    }

    return c[1][1] + (int)b;
  }
};

int bar(int n) {
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

extern void assert(int) throw() __attribute__((__noreturn__));
void unreachable_call() {
#pragma omp target
    assert(0);
}

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+static.+347}}_worker()
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

// CHECK: define {{.*}}void [[T4:@__omp_offloading_.+static.+l347]](i[[SZ]]
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

// CHECK-LABEL: define {{.*}}void {{@__omp_offloading_.+S1.+l365}}_worker()
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

// CHECK: define {{.*}}void [[T5:@__omp_offloading_.+S1.+l365]](
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

// CHECK: define{{ hidden | }}i32 [[BAZ]](i32 [[F:%.*]], double* nonnull align {{[0-9]+}} dereferenceable{{.*}})
// CHECK: alloca i32,
// CHECK: [[LOCAL_F_PTR:%.+]] = alloca i32,
// CHECK: [[ZERO_ADDR:%.+]] = alloca i32,
// CHECK: [[BND_ZERO_ADDR:%.+]] = alloca i32,
// CHECK: store i32 0, i32* [[BND_ZERO_ADDR]]
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
// CHECK: call void [[OUTLINED:@.+]](i32* [[ZERO_ADDR]], i32* [[BND_ZERO_ADDR]], i32* [[F_PTR]], double* %{{.+}})
// CHECK: call void @__kmpc_end_serialized_parallel(%struct.ident_t* [[UNKNOWN]], i32 [[GTID]])
// CHECK: br label

// CHECK: call void @__kmpc_kernel_prepare_parallel(i8* bitcast (void (i16, i32)* @{{.+}} to i8*))
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

// CHECK: define {{.*}}void {{@__omp_offloading_.+unreachable_call.+l399}}()
// CHECK: call void @{{.*}}assert{{.*}}(i32 0)
// CHECK: unreachable
// CHECK: call void @__kmpc_kernel_deinit(i16 1)
// CHECK: ret void

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
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], ptr [[PTR1:%.*]], ptr nonnull align 8 dereferenceable(8) [[PTR2:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[PTR1_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[PTR2_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[CAPTURED_VARS_ADDRS:%.*]] = alloca [2 x ptr], align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[PTR1]], ptr [[PTR1_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[PTR2]], ptr [[PTR2_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR2_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1:[0-9]+]])
// CHECK1-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i64 0, i64 0
// CHECK1-NEXT:    store ptr [[PTR1_ADDR]], ptr [[TMP3]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i64 0, i64 1
// CHECK1-NEXT:    store ptr [[TMP0]], ptr [[TMP4]], align 8
// CHECK1-NEXT:    call void @__kmpc_parallel_51(ptr @[[GLOB1]], i32 [[TMP2]], i32 1, i32 2, i32 -1, ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_omp_outlined, ptr null, ptr [[CAPTURED_VARS_ADDRS]], i64 2)
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_omp_outlined
// CHECK1-SAME: (ptr noalias [[DOTGLOBAL_TID_:%.*]], ptr noalias [[DOTBOUND_TID_:%.*]], ptr nonnull align 8 dereferenceable(8) [[PTR1:%.*]], ptr nonnull align 8 dereferenceable(8) [[PTR2:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[PTR1_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[PTR2_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[PTR1]], ptr [[PTR1_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[PTR2]], ptr [[PTR2_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR1_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PTR2_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[TMP1]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 4
// CHECK1-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP0]], align 8
// CHECK1-NEXT:    store i32 [[TMP3]], ptr [[TMP4]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l39
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]]) #[[ATTR4:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l39_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l47
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], i64 [[AA:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[AA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[AA]], ptr [[AA_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l47_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP1:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[CONV:%.*]] = sext i16 [[TMP1]] to i32
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[CONV]], 1
// CHECK1-NEXT:    [[CONV1:%.*]] = trunc i32 [[ADD]] to i16
// CHECK1-NEXT:    store i16 [[CONV1]], ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[TMP2:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[CONV2:%.*]] = sext i16 [[TMP2]] to i32
// CHECK1-NEXT:    [[ADD3:%.*]] = add nsw i32 [[CONV2]], 2
// CHECK1-NEXT:    [[CONV4:%.*]] = trunc i32 [[ADD3]] to i16
// CHECK1-NEXT:    store i16 [[CONV4]], ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l53
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], i64 [[A:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]], i64 [[VLA:%.*]], ptr nonnull align 4 dereferenceable(4) [[BN:%.*]], ptr nonnull align 8 dereferenceable(400) [[C:%.*]], i64 [[VLA1:%.*]], i64 [[VLA3:%.*]], ptr nonnull align 8 dereferenceable(8) [[CN:%.*]], ptr nonnull align 8 dereferenceable(16) [[D:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[VLA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[BN_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[C_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[VLA_ADDR2:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[VLA_ADDR4:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[CN_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[D_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA]], ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[BN]], ptr [[BN_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[C]], ptr [[C_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA1]], ptr [[VLA_ADDR2]], align 8
// CHECK1-NEXT:    store i64 [[VLA3]], ptr [[VLA_ADDR4]], align 8
// CHECK1-NEXT:    store ptr [[CN]], ptr [[CN_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[D]], ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load i64, ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[BN_ADDR]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[C_ADDR]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = load i64, ptr [[VLA_ADDR2]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = load i64, ptr [[VLA_ADDR4]], align 8
// CHECK1-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[CN_ADDR]], align 8
// CHECK1-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[D_ADDR]], align 8
// CHECK1-NEXT:    [[TMP8:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l53_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP8]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP9:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP9]], 1
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x float], ptr [[TMP0]], i64 0, i64 2
// CHECK1-NEXT:    [[TMP10:%.*]] = load float, ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    [[CONV:%.*]] = fpext float [[TMP10]] to double
// CHECK1-NEXT:    [[ADD5:%.*]] = fadd double [[CONV]], 1.000000e+00
// CHECK1-NEXT:    [[CONV6:%.*]] = fptrunc double [[ADD5]] to float
// CHECK1-NEXT:    store float [[CONV6]], ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds float, ptr [[TMP2]], i64 3
// CHECK1-NEXT:    [[TMP11:%.*]] = load float, ptr [[ARRAYIDX7]], align 4
// CHECK1-NEXT:    [[CONV8:%.*]] = fpext float [[TMP11]] to double
// CHECK1-NEXT:    [[ADD9:%.*]] = fadd double [[CONV8]], 1.000000e+00
// CHECK1-NEXT:    [[CONV10:%.*]] = fptrunc double [[ADD9]] to float
// CHECK1-NEXT:    store float [[CONV10]], ptr [[ARRAYIDX7]], align 4
// CHECK1-NEXT:    [[ARRAYIDX11:%.*]] = getelementptr inbounds [5 x [10 x double]], ptr [[TMP3]], i64 0, i64 1
// CHECK1-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds [10 x double], ptr [[ARRAYIDX11]], i64 0, i64 2
// CHECK1-NEXT:    [[TMP12:%.*]] = load double, ptr [[ARRAYIDX12]], align 8
// CHECK1-NEXT:    [[ADD13:%.*]] = fadd double [[TMP12]], 1.000000e+00
// CHECK1-NEXT:    store double [[ADD13]], ptr [[ARRAYIDX12]], align 8
// CHECK1-NEXT:    [[TMP13:%.*]] = mul nsw i64 1, [[TMP5]]
// CHECK1-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds double, ptr [[TMP6]], i64 [[TMP13]]
// CHECK1-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds double, ptr [[ARRAYIDX14]], i64 3
// CHECK1-NEXT:    [[TMP14:%.*]] = load double, ptr [[ARRAYIDX15]], align 8
// CHECK1-NEXT:    [[ADD16:%.*]] = fadd double [[TMP14]], 1.000000e+00
// CHECK1-NEXT:    store double [[ADD16]], ptr [[ARRAYIDX15]], align 8
// CHECK1-NEXT:    [[X:%.*]] = getelementptr inbounds [[STRUCT_TT:%.*]], ptr [[TMP7]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP15:%.*]] = load i64, ptr [[X]], align 8
// CHECK1-NEXT:    [[ADD17:%.*]] = add nsw i64 [[TMP15]], 1
// CHECK1-NEXT:    store i64 [[ADD17]], ptr [[X]], align 8
// CHECK1-NEXT:    [[Y:%.*]] = getelementptr inbounds [[STRUCT_TT]], ptr [[TMP7]], i32 0, i32 1
// CHECK1-NEXT:    [[TMP16:%.*]] = load i8, ptr [[Y]], align 8
// CHECK1-NEXT:    [[CONV18:%.*]] = sext i8 [[TMP16]] to i32
// CHECK1-NEXT:    [[ADD19:%.*]] = add nsw i32 [[CONV18]], 1
// CHECK1-NEXT:    [[CONV20:%.*]] = trunc i32 [[ADD19]] to i8
// CHECK1-NEXT:    store i8 [[CONV20]], ptr [[Y]], align 8
// CHECK1-NEXT:    [[CALL:%.*]] = call nonnull align 8 dereferenceable(8) ptr @_ZN2TTIxcEixEi(ptr nonnull align 8 dereferenceable(16) [[TMP7]], i32 0) #[[ATTR10:[0-9]+]]
// CHECK1-NEXT:    [[TMP17:%.*]] = load i64, ptr [[CALL]], align 8
// CHECK1-NEXT:    [[ADD21:%.*]] = add nsw i64 [[TMP17]], 1
// CHECK1-NEXT:    store i64 [[ADD21]], ptr [[CALL]], align 8
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_ZN2TTIxcEixEi
// CHECK1-SAME: (ptr nonnull align 8 dereferenceable(16) [[THIS:%.*]], i32 [[I:%.*]]) #[[ATTR5:[0-9]+]] comdat align 2 {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[I_ADDR:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    store i32 [[I]], ptr [[I_ADDR]], align 4
// CHECK1-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[X:%.*]] = getelementptr inbounds [[STRUCT_TT:%.*]], ptr [[THIS1]], i32 0, i32 0
// CHECK1-NEXT:    ret ptr [[X]]
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZL7fstatici_l90
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], i64 [[A:%.*]], i64 [[AA:%.*]], i64 [[AAA:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[AA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[AAA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[AA]], ptr [[AA_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[AAA]], ptr [[AAA_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZL7fstatici_l90_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP2]], 1
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[TMP3:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[CONV:%.*]] = sext i16 [[TMP3]] to i32
// CHECK1-NEXT:    [[ADD1:%.*]] = add nsw i32 [[CONV]], 1
// CHECK1-NEXT:    [[CONV2:%.*]] = trunc i32 [[ADD1]] to i16
// CHECK1-NEXT:    store i16 [[CONV2]], ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[TMP4:%.*]] = load i8, ptr [[AAA_ADDR]], align 1
// CHECK1-NEXT:    [[CONV3:%.*]] = sext i8 [[TMP4]] to i32
// CHECK1-NEXT:    [[ADD4:%.*]] = add nsw i32 [[CONV3]], 1
// CHECK1-NEXT:    [[CONV5:%.*]] = trunc i32 [[ADD4]] to i8
// CHECK1-NEXT:    store i8 [[CONV5]], ptr [[AAA_ADDR]], align 1
// CHECK1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x i32], ptr [[TMP0]], i64 0, i64 2
// CHECK1-NEXT:    [[TMP5:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP5]], 1
// CHECK1-NEXT:    store i32 [[ADD6]], ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZN2S12r1Ei_l108
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], ptr [[THIS:%.*]], i64 [[B:%.*]], i64 [[VLA:%.*]], i64 [[VLA1:%.*]], ptr nonnull align 2 dereferenceable(2) [[C:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[B_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[VLA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[VLA_ADDR2:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[C_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[B]], ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA]], ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[VLA1]], ptr [[VLA_ADDR2]], align 8
// CHECK1-NEXT:    store ptr [[C]], ptr [[C_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load i64, ptr [[VLA_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load i64, ptr [[VLA_ADDR2]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[C_ADDR]], align 8
// CHECK1-NEXT:    [[TMP4:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZN2S12r1Ei_l108_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP4]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP5:%.*]] = load i32, ptr [[B_ADDR]], align 4
// CHECK1-NEXT:    [[CONV:%.*]] = sitofp i32 [[TMP5]] to double
// CHECK1-NEXT:    [[ADD:%.*]] = fadd double [[CONV]], 1.500000e+00
// CHECK1-NEXT:    [[A:%.*]] = getelementptr inbounds [[STRUCT_S1:%.*]], ptr [[TMP0]], i32 0, i32 0
// CHECK1-NEXT:    store double [[ADD]], ptr [[A]], align 8
// CHECK1-NEXT:    [[A3:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP6:%.*]] = load double, ptr [[A3]], align 8
// CHECK1-NEXT:    [[INC:%.*]] = fadd double [[TMP6]], 1.000000e+00
// CHECK1-NEXT:    store double [[INC]], ptr [[A3]], align 8
// CHECK1-NEXT:    [[CONV4:%.*]] = fptosi double [[INC]] to i16
// CHECK1-NEXT:    [[TMP7:%.*]] = mul nsw i64 1, [[TMP2]]
// CHECK1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i16, ptr [[TMP3]], i64 [[TMP7]]
// CHECK1-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds i16, ptr [[ARRAYIDX]], i64 1
// CHECK1-NEXT:    store i16 [[CONV4]], ptr [[ARRAYIDX5]], align 2
// CHECK1-NEXT:    [[A6:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK1-NEXT:    [[TMP8:%.*]] = load double, ptr [[A6]], align 8
// CHECK1-NEXT:    [[CONV7:%.*]] = fptosi double [[TMP8]] to i32
// CHECK1-NEXT:    [[A8:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK1-NEXT:    [[CALL:%.*]] = call i32 @_Z3baziRd(i32 [[CONV7]], ptr nonnull align 8 dereferenceable(8) [[A8]]) #[[ATTR10]]
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_Z3baziRd
// CHECK1-SAME: (i32 [[F1:%.*]], ptr nonnull align 8 dereferenceable(8) [[A:%.*]]) #[[ATTR5]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[CAPTURED_VARS_ADDRS:%.*]] = alloca [2 x ptr], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1]])
// CHECK1-NEXT:    [[F:%.*]] = call align 8 ptr @__kmpc_alloc_shared(i64 4)
// CHECK1-NEXT:    store i32 [[F1]], ptr [[F]], align 4
// CHECK1-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i64 0, i64 0
// CHECK1-NEXT:    store ptr [[F]], ptr [[TMP2]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i64 0, i64 1
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[TMP3]], align 8
// CHECK1-NEXT:    call void @__kmpc_parallel_51(ptr @[[GLOB1]], i32 [[TMP0]], i32 1, i32 -1, i32 -1, ptr @_Z3baziRd_omp_outlined, ptr @_Z3baziRd_omp_outlined_wrapper, ptr [[CAPTURED_VARS_ADDRS]], i64 2)
// CHECK1-NEXT:    [[TMP4:%.*]] = load i32, ptr [[F]], align 4
// CHECK1-NEXT:    call void @__kmpc_free_shared(ptr [[F]], i64 4)
// CHECK1-NEXT:    ret i32 [[TMP4]]
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z16unreachable_callv_l142
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z16unreachable_callv_l142_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    call void @_Z6asserti(i32 0) #[[ATTR11:[0-9]+]]
// CHECK1-NEXT:    unreachable
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
// CHECK1:       1:
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9ftemplateIiET_i_l74
// CHECK1-SAME: (ptr noalias [[DYN_PTR:%.*]], i64 [[A:%.*]], i64 [[AA:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]]) #[[ATTR4]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[AA_ADDR:%.*]] = alloca i64, align 8
// CHECK1-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store i64 [[AA]], ptr [[AA_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9ftemplateIiET_i_l74_kernel_environment, ptr [[DYN_PTR]])
// CHECK1-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK1-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK1:       user_code.entry:
// CHECK1-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP2]], 1
// CHECK1-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK1-NEXT:    [[TMP3:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[CONV:%.*]] = sext i16 [[TMP3]] to i32
// CHECK1-NEXT:    [[ADD1:%.*]] = add nsw i32 [[CONV]], 1
// CHECK1-NEXT:    [[CONV2:%.*]] = trunc i32 [[ADD1]] to i16
// CHECK1-NEXT:    store i16 [[CONV2]], ptr [[AA_ADDR]], align 2
// CHECK1-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x i32], ptr [[TMP0]], i64 0, i64 2
// CHECK1-NEXT:    [[TMP4:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 1
// CHECK1-NEXT:    store i32 [[ADD3]], ptr [[ARRAYIDX]], align 4
// CHECK1-NEXT:    call void @__kmpc_target_deinit()
// CHECK1-NEXT:    ret void
// CHECK1:       worker.exit:
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_Z3baziRd_omp_outlined
// CHECK1-SAME: (ptr noalias [[DOTGLOBAL_TID_:%.*]], ptr noalias [[DOTBOUND_TID_:%.*]], ptr nonnull align 4 dereferenceable(4) [[F:%.*]], ptr nonnull align 8 dereferenceable(8) [[A:%.*]]) #[[ATTR1]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[F_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    [[TMP:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 8
// CHECK1-NEXT:    store ptr [[F]], ptr [[F_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[F_ADDR]], align 8
// CHECK1-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[A_ADDR]], align 8
// CHECK1-NEXT:    store ptr [[TMP1]], ptr [[TMP]], align 8
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[TMP]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = load double, ptr [[TMP2]], align 8
// CHECK1-NEXT:    [[ADD:%.*]] = fadd double 2.000000e+00, [[TMP3]]
// CHECK1-NEXT:    [[CONV:%.*]] = fptosi double [[ADD]] to i32
// CHECK1-NEXT:    store i32 [[CONV]], ptr [[TMP0]], align 4
// CHECK1-NEXT:    ret void
//
//
// CHECK1-LABEL: define {{[^@]+}}@_Z3baziRd_omp_outlined_wrapper
// CHECK1-SAME: (i16 zeroext [[TMP0:%.*]], i32 [[TMP1:%.*]]) #[[ATTR8:[0-9]+]] {
// CHECK1-NEXT:  entry:
// CHECK1-NEXT:    [[DOTADDR:%.*]] = alloca i16, align 2
// CHECK1-NEXT:    [[DOTADDR1:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[DOTZERO_ADDR:%.*]] = alloca i32, align 4
// CHECK1-NEXT:    [[GLOBAL_ARGS:%.*]] = alloca ptr, align 8
// CHECK1-NEXT:    store i16 [[TMP0]], ptr [[DOTADDR]], align 2
// CHECK1-NEXT:    store i32 [[TMP1]], ptr [[DOTADDR1]], align 4
// CHECK1-NEXT:    store i32 0, ptr [[DOTZERO_ADDR]], align 4
// CHECK1-NEXT:    call void @__kmpc_get_shared_variables(ptr [[GLOBAL_ARGS]])
// CHECK1-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[GLOBAL_ARGS]], align 8
// CHECK1-NEXT:    [[TMP3:%.*]] = getelementptr inbounds ptr, ptr [[TMP2]], i64 0
// CHECK1-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP3]], align 8
// CHECK1-NEXT:    [[TMP5:%.*]] = getelementptr inbounds ptr, ptr [[TMP2]], i64 1
// CHECK1-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[TMP5]], align 8
// CHECK1-NEXT:    call void @_Z3baziRd_omp_outlined(ptr [[DOTADDR1]], ptr [[DOTZERO_ADDR]], ptr [[TMP4]], ptr [[TMP6]]) #[[ATTR2:[0-9]+]]
// CHECK1-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], ptr [[PTR1:%.*]], ptr nonnull align 4 dereferenceable(4) [[PTR2:%.*]]) #[[ATTR0:[0-9]+]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[PTR1_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[PTR2_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[CAPTURED_VARS_ADDRS:%.*]] = alloca [2 x ptr], align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[PTR1]], ptr [[PTR1_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[PTR2]], ptr [[PTR2_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR2_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP2:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1:[0-9]+]])
// CHECK2-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i32 0, i32 0
// CHECK2-NEXT:    store ptr [[PTR1_ADDR]], ptr [[TMP3]], align 4
// CHECK2-NEXT:    [[TMP4:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i32 0, i32 1
// CHECK2-NEXT:    store ptr [[TMP0]], ptr [[TMP4]], align 4
// CHECK2-NEXT:    call void @__kmpc_parallel_51(ptr @[[GLOB1]], i32 [[TMP2]], i32 1, i32 2, i32 -1, ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_omp_outlined, ptr null, ptr [[CAPTURED_VARS_ADDRS]], i32 2)
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9targetBarPiS__l25_omp_outlined
// CHECK2-SAME: (ptr noalias [[DOTGLOBAL_TID_:%.*]], ptr noalias [[DOTBOUND_TID_:%.*]], ptr nonnull align 4 dereferenceable(4) [[PTR1:%.*]], ptr nonnull align 4 dereferenceable(4) [[PTR2:%.*]]) #[[ATTR1:[0-9]+]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[PTR1_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[PTR2_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 4
// CHECK2-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 4
// CHECK2-NEXT:    store ptr [[PTR1]], ptr [[PTR1_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[PTR2]], ptr [[PTR2_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR1_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[PTR2_ADDR]], align 4
// CHECK2-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[TMP1]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load i32, ptr [[TMP2]], align 4
// CHECK2-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP0]], align 4
// CHECK2-NEXT:    store i32 [[TMP3]], ptr [[TMP4]], align 4
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l39
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]]) #[[ATTR4:[0-9]+]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l39_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l47
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], i32 [[AA:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[AA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[AA]], ptr [[AA_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l47_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP1:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[CONV:%.*]] = sext i16 [[TMP1]] to i32
// CHECK2-NEXT:    [[ADD:%.*]] = add nsw i32 [[CONV]], 1
// CHECK2-NEXT:    [[CONV1:%.*]] = trunc i32 [[ADD]] to i16
// CHECK2-NEXT:    store i16 [[CONV1]], ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[TMP2:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[CONV2:%.*]] = sext i16 [[TMP2]] to i32
// CHECK2-NEXT:    [[ADD3:%.*]] = add nsw i32 [[CONV2]], 2
// CHECK2-NEXT:    [[CONV4:%.*]] = trunc i32 [[ADD3]] to i16
// CHECK2-NEXT:    store i16 [[CONV4]], ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l53
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], i32 [[A:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]], i32 [[VLA:%.*]], ptr nonnull align 4 dereferenceable(4) [[BN:%.*]], ptr nonnull align 8 dereferenceable(400) [[C:%.*]], i32 [[VLA1:%.*]], i32 [[VLA3:%.*]], ptr nonnull align 8 dereferenceable(8) [[CN:%.*]], ptr nonnull align 8 dereferenceable(16) [[D:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[VLA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[BN_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[C_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[VLA_ADDR2:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[VLA_ADDR4:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[CN_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[D_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[VLA]], ptr [[VLA_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[BN]], ptr [[BN_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[C]], ptr [[C_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[VLA1]], ptr [[VLA_ADDR2]], align 4
// CHECK2-NEXT:    store i32 [[VLA3]], ptr [[VLA_ADDR4]], align 4
// CHECK2-NEXT:    store ptr [[CN]], ptr [[CN_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[D]], ptr [[D_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = load i32, ptr [[VLA_ADDR]], align 4
// CHECK2-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[BN_ADDR]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[C_ADDR]], align 4
// CHECK2-NEXT:    [[TMP4:%.*]] = load i32, ptr [[VLA_ADDR2]], align 4
// CHECK2-NEXT:    [[TMP5:%.*]] = load i32, ptr [[VLA_ADDR4]], align 4
// CHECK2-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[CN_ADDR]], align 4
// CHECK2-NEXT:    [[TMP7:%.*]] = load ptr, ptr [[D_ADDR]], align 4
// CHECK2-NEXT:    [[TMP8:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z3fooi_l53_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP8]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP9:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP9]], 1
// CHECK2-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x float], ptr [[TMP0]], i32 0, i32 2
// CHECK2-NEXT:    [[TMP10:%.*]] = load float, ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    [[CONV:%.*]] = fpext float [[TMP10]] to double
// CHECK2-NEXT:    [[ADD5:%.*]] = fadd double [[CONV]], 1.000000e+00
// CHECK2-NEXT:    [[CONV6:%.*]] = fptrunc double [[ADD5]] to float
// CHECK2-NEXT:    store float [[CONV6]], ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    [[ARRAYIDX7:%.*]] = getelementptr inbounds float, ptr [[TMP2]], i32 3
// CHECK2-NEXT:    [[TMP11:%.*]] = load float, ptr [[ARRAYIDX7]], align 4
// CHECK2-NEXT:    [[CONV8:%.*]] = fpext float [[TMP11]] to double
// CHECK2-NEXT:    [[ADD9:%.*]] = fadd double [[CONV8]], 1.000000e+00
// CHECK2-NEXT:    [[CONV10:%.*]] = fptrunc double [[ADD9]] to float
// CHECK2-NEXT:    store float [[CONV10]], ptr [[ARRAYIDX7]], align 4
// CHECK2-NEXT:    [[ARRAYIDX11:%.*]] = getelementptr inbounds [5 x [10 x double]], ptr [[TMP3]], i32 0, i32 1
// CHECK2-NEXT:    [[ARRAYIDX12:%.*]] = getelementptr inbounds [10 x double], ptr [[ARRAYIDX11]], i32 0, i32 2
// CHECK2-NEXT:    [[TMP12:%.*]] = load double, ptr [[ARRAYIDX12]], align 8
// CHECK2-NEXT:    [[ADD13:%.*]] = fadd double [[TMP12]], 1.000000e+00
// CHECK2-NEXT:    store double [[ADD13]], ptr [[ARRAYIDX12]], align 8
// CHECK2-NEXT:    [[TMP13:%.*]] = mul nsw i32 1, [[TMP5]]
// CHECK2-NEXT:    [[ARRAYIDX14:%.*]] = getelementptr inbounds double, ptr [[TMP6]], i32 [[TMP13]]
// CHECK2-NEXT:    [[ARRAYIDX15:%.*]] = getelementptr inbounds double, ptr [[ARRAYIDX14]], i32 3
// CHECK2-NEXT:    [[TMP14:%.*]] = load double, ptr [[ARRAYIDX15]], align 8
// CHECK2-NEXT:    [[ADD16:%.*]] = fadd double [[TMP14]], 1.000000e+00
// CHECK2-NEXT:    store double [[ADD16]], ptr [[ARRAYIDX15]], align 8
// CHECK2-NEXT:    [[X:%.*]] = getelementptr inbounds [[STRUCT_TT:%.*]], ptr [[TMP7]], i32 0, i32 0
// CHECK2-NEXT:    [[TMP15:%.*]] = load i64, ptr [[X]], align 8
// CHECK2-NEXT:    [[ADD17:%.*]] = add nsw i64 [[TMP15]], 1
// CHECK2-NEXT:    store i64 [[ADD17]], ptr [[X]], align 8
// CHECK2-NEXT:    [[Y:%.*]] = getelementptr inbounds [[STRUCT_TT]], ptr [[TMP7]], i32 0, i32 1
// CHECK2-NEXT:    [[TMP16:%.*]] = load i8, ptr [[Y]], align 8
// CHECK2-NEXT:    [[CONV18:%.*]] = sext i8 [[TMP16]] to i32
// CHECK2-NEXT:    [[ADD19:%.*]] = add nsw i32 [[CONV18]], 1
// CHECK2-NEXT:    [[CONV20:%.*]] = trunc i32 [[ADD19]] to i8
// CHECK2-NEXT:    store i8 [[CONV20]], ptr [[Y]], align 8
// CHECK2-NEXT:    [[CALL:%.*]] = call nonnull align 8 dereferenceable(8) ptr @_ZN2TTIxcEixEi(ptr nonnull align 8 dereferenceable(16) [[TMP7]], i32 0) #[[ATTR10:[0-9]+]]
// CHECK2-NEXT:    [[TMP17:%.*]] = load i64, ptr [[CALL]], align 8
// CHECK2-NEXT:    [[ADD21:%.*]] = add nsw i64 [[TMP17]], 1
// CHECK2-NEXT:    store i64 [[ADD21]], ptr [[CALL]], align 8
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@_ZN2TTIxcEixEi
// CHECK2-SAME: (ptr nonnull align 8 dereferenceable(16) [[THIS:%.*]], i32 [[I:%.*]]) #[[ATTR5:[0-9]+]] comdat align 2 {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[I_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[I]], ptr [[I_ADDR]], align 4
// CHECK2-NEXT:    [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 4
// CHECK2-NEXT:    [[X:%.*]] = getelementptr inbounds [[STRUCT_TT:%.*]], ptr [[THIS1]], i32 0, i32 0
// CHECK2-NEXT:    ret ptr [[X]]
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZL7fstatici_l90
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], i32 [[A:%.*]], i32 [[AA:%.*]], i32 [[AAA:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[AA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[AAA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[AA]], ptr [[AA_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[AAA]], ptr [[AAA_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZL7fstatici_l90_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP2]], 1
// CHECK2-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[CONV:%.*]] = sext i16 [[TMP3]] to i32
// CHECK2-NEXT:    [[ADD1:%.*]] = add nsw i32 [[CONV]], 1
// CHECK2-NEXT:    [[CONV2:%.*]] = trunc i32 [[ADD1]] to i16
// CHECK2-NEXT:    store i16 [[CONV2]], ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[TMP4:%.*]] = load i8, ptr [[AAA_ADDR]], align 1
// CHECK2-NEXT:    [[CONV3:%.*]] = sext i8 [[TMP4]] to i32
// CHECK2-NEXT:    [[ADD4:%.*]] = add nsw i32 [[CONV3]], 1
// CHECK2-NEXT:    [[CONV5:%.*]] = trunc i32 [[ADD4]] to i8
// CHECK2-NEXT:    store i8 [[CONV5]], ptr [[AAA_ADDR]], align 1
// CHECK2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x i32], ptr [[TMP0]], i32 0, i32 2
// CHECK2-NEXT:    [[TMP5:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    [[ADD6:%.*]] = add nsw i32 [[TMP5]], 1
// CHECK2-NEXT:    store i32 [[ADD6]], ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZN2S12r1Ei_l108
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], ptr [[THIS:%.*]], i32 [[B:%.*]], i32 [[VLA:%.*]], i32 [[VLA1:%.*]], ptr nonnull align 2 dereferenceable(2) [[C:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[THIS_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[B_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[VLA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[VLA_ADDR2:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[C_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[THIS]], ptr [[THIS_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[B]], ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[VLA]], ptr [[VLA_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[VLA1]], ptr [[VLA_ADDR2]], align 4
// CHECK2-NEXT:    store ptr [[C]], ptr [[C_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[THIS_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = load i32, ptr [[VLA_ADDR]], align 4
// CHECK2-NEXT:    [[TMP2:%.*]] = load i32, ptr [[VLA_ADDR2]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load ptr, ptr [[C_ADDR]], align 4
// CHECK2-NEXT:    [[TMP4:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__ZN2S12r1Ei_l108_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP4]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP5:%.*]] = load i32, ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[CONV:%.*]] = sitofp i32 [[TMP5]] to double
// CHECK2-NEXT:    [[ADD:%.*]] = fadd double [[CONV]], 1.500000e+00
// CHECK2-NEXT:    [[A:%.*]] = getelementptr inbounds [[STRUCT_S1:%.*]], ptr [[TMP0]], i32 0, i32 0
// CHECK2-NEXT:    store double [[ADD]], ptr [[A]], align 8
// CHECK2-NEXT:    [[A3:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK2-NEXT:    [[TMP6:%.*]] = load double, ptr [[A3]], align 8
// CHECK2-NEXT:    [[INC:%.*]] = fadd double [[TMP6]], 1.000000e+00
// CHECK2-NEXT:    store double [[INC]], ptr [[A3]], align 8
// CHECK2-NEXT:    [[CONV4:%.*]] = fptosi double [[INC]] to i16
// CHECK2-NEXT:    [[TMP7:%.*]] = mul nsw i32 1, [[TMP2]]
// CHECK2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i16, ptr [[TMP3]], i32 [[TMP7]]
// CHECK2-NEXT:    [[ARRAYIDX5:%.*]] = getelementptr inbounds i16, ptr [[ARRAYIDX]], i32 1
// CHECK2-NEXT:    store i16 [[CONV4]], ptr [[ARRAYIDX5]], align 2
// CHECK2-NEXT:    [[A6:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK2-NEXT:    [[TMP8:%.*]] = load double, ptr [[A6]], align 8
// CHECK2-NEXT:    [[CONV7:%.*]] = fptosi double [[TMP8]] to i32
// CHECK2-NEXT:    [[A8:%.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[TMP0]], i32 0, i32 0
// CHECK2-NEXT:    [[CALL:%.*]] = call i32 @_Z3baziRd(i32 [[CONV7]], ptr nonnull align 8 dereferenceable(8) [[A8]]) #[[ATTR10]]
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@_Z3baziRd
// CHECK2-SAME: (i32 [[F1:%.*]], ptr nonnull align 8 dereferenceable(8) [[A:%.*]]) #[[ATTR5]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[CAPTURED_VARS_ADDRS:%.*]] = alloca [2 x ptr], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_global_thread_num(ptr @[[GLOB1]])
// CHECK2-NEXT:    [[F:%.*]] = call align 8 ptr @__kmpc_alloc_shared(i32 4)
// CHECK2-NEXT:    store i32 [[F1]], ptr [[F]], align 4
// CHECK2-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[TMP2:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i32 0, i32 0
// CHECK2-NEXT:    store ptr [[F]], ptr [[TMP2]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [2 x ptr], ptr [[CAPTURED_VARS_ADDRS]], i32 0, i32 1
// CHECK2-NEXT:    store ptr [[TMP1]], ptr [[TMP3]], align 4
// CHECK2-NEXT:    call void @__kmpc_parallel_51(ptr @[[GLOB1]], i32 [[TMP0]], i32 1, i32 -1, i32 -1, ptr @_Z3baziRd_omp_outlined, ptr @_Z3baziRd_omp_outlined_wrapper, ptr [[CAPTURED_VARS_ADDRS]], i32 2)
// CHECK2-NEXT:    [[TMP4:%.*]] = load i32, ptr [[F]], align 4
// CHECK2-NEXT:    call void @__kmpc_free_shared(ptr [[F]], i32 4)
// CHECK2-NEXT:    ret i32 [[TMP4]]
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z16unreachable_callv_l142
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z16unreachable_callv_l142_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP0]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    call void @_Z6asserti(i32 0) #[[ATTR11:[0-9]+]]
// CHECK2-NEXT:    unreachable
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
// CHECK2:       1:
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9ftemplateIiET_i_l74
// CHECK2-SAME: (ptr noalias [[DYN_PTR:%.*]], i32 [[A:%.*]], i32 [[AA:%.*]], ptr nonnull align 4 dereferenceable(40) [[B:%.*]]) #[[ATTR4]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DYN_PTR_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[AA_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[B_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DYN_PTR]], ptr [[DYN_PTR_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[A]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    store i32 [[AA]], ptr [[AA_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[B]], ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[B_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = call i32 @__kmpc_target_init(ptr @{{__omp_offloading_[0-9a-z]+_[0-9a-z]+}}__Z9ftemplateIiET_i_l74_kernel_environment, ptr [[DYN_PTR]])
// CHECK2-NEXT:    [[EXEC_USER_CODE:%.*]] = icmp eq i32 [[TMP1]], -1
// CHECK2-NEXT:    br i1 [[EXEC_USER_CODE]], label [[USER_CODE_ENTRY:%.*]], label [[WORKER_EXIT:%.*]]
// CHECK2:       user_code.entry:
// CHECK2-NEXT:    [[TMP2:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP2]], 1
// CHECK2-NEXT:    store i32 [[ADD]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load i16, ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[CONV:%.*]] = sext i16 [[TMP3]] to i32
// CHECK2-NEXT:    [[ADD1:%.*]] = add nsw i32 [[CONV]], 1
// CHECK2-NEXT:    [[CONV2:%.*]] = trunc i32 [[ADD1]] to i16
// CHECK2-NEXT:    store i16 [[CONV2]], ptr [[AA_ADDR]], align 2
// CHECK2-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds [10 x i32], ptr [[TMP0]], i32 0, i32 2
// CHECK2-NEXT:    [[TMP4:%.*]] = load i32, ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    [[ADD3:%.*]] = add nsw i32 [[TMP4]], 1
// CHECK2-NEXT:    store i32 [[ADD3]], ptr [[ARRAYIDX]], align 4
// CHECK2-NEXT:    call void @__kmpc_target_deinit()
// CHECK2-NEXT:    ret void
// CHECK2:       worker.exit:
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@_Z3baziRd_omp_outlined
// CHECK2-SAME: (ptr noalias [[DOTGLOBAL_TID_:%.*]], ptr noalias [[DOTBOUND_TID_:%.*]], ptr nonnull align 4 dereferenceable(4) [[F:%.*]], ptr nonnull align 8 dereferenceable(8) [[A:%.*]]) #[[ATTR1]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DOTGLOBAL_TID__ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[DOTBOUND_TID__ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[F_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[A_ADDR:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    [[TMP:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store ptr [[DOTGLOBAL_TID_]], ptr [[DOTGLOBAL_TID__ADDR]], align 4
// CHECK2-NEXT:    store ptr [[DOTBOUND_TID_]], ptr [[DOTBOUND_TID__ADDR]], align 4
// CHECK2-NEXT:    store ptr [[F]], ptr [[F_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[A]], ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[F_ADDR]], align 4
// CHECK2-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[A_ADDR]], align 4
// CHECK2-NEXT:    store ptr [[TMP1]], ptr [[TMP]], align 4
// CHECK2-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[TMP]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = load double, ptr [[TMP2]], align 8
// CHECK2-NEXT:    [[ADD:%.*]] = fadd double 2.000000e+00, [[TMP3]]
// CHECK2-NEXT:    [[CONV:%.*]] = fptosi double [[ADD]] to i32
// CHECK2-NEXT:    store i32 [[CONV]], ptr [[TMP0]], align 4
// CHECK2-NEXT:    ret void
//
//
// CHECK2-LABEL: define {{[^@]+}}@_Z3baziRd_omp_outlined_wrapper
// CHECK2-SAME: (i16 zeroext [[TMP0:%.*]], i32 [[TMP1:%.*]]) #[[ATTR8:[0-9]+]] {
// CHECK2-NEXT:  entry:
// CHECK2-NEXT:    [[DOTADDR:%.*]] = alloca i16, align 2
// CHECK2-NEXT:    [[DOTADDR1:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[DOTZERO_ADDR:%.*]] = alloca i32, align 4
// CHECK2-NEXT:    [[GLOBAL_ARGS:%.*]] = alloca ptr, align 4
// CHECK2-NEXT:    store i16 [[TMP0]], ptr [[DOTADDR]], align 2
// CHECK2-NEXT:    store i32 [[TMP1]], ptr [[DOTADDR1]], align 4
// CHECK2-NEXT:    store i32 0, ptr [[DOTZERO_ADDR]], align 4
// CHECK2-NEXT:    call void @__kmpc_get_shared_variables(ptr [[GLOBAL_ARGS]])
// CHECK2-NEXT:    [[TMP2:%.*]] = load ptr, ptr [[GLOBAL_ARGS]], align 4
// CHECK2-NEXT:    [[TMP3:%.*]] = getelementptr inbounds ptr, ptr [[TMP2]], i32 0
// CHECK2-NEXT:    [[TMP4:%.*]] = load ptr, ptr [[TMP3]], align 4
// CHECK2-NEXT:    [[TMP5:%.*]] = getelementptr inbounds ptr, ptr [[TMP2]], i32 1
// CHECK2-NEXT:    [[TMP6:%.*]] = load ptr, ptr [[TMP5]], align 4
// CHECK2-NEXT:    call void @_Z3baziRd_omp_outlined(ptr [[DOTADDR1]], ptr [[DOTZERO_ADDR]], ptr [[TMP4]], ptr [[TMP6]]) #[[ATTR2:[0-9]+]]
// CHECK2-NEXT:    ret void
//
