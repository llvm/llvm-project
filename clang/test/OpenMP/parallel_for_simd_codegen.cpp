// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// REQUIRES: x86-registered-target
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

long long get_val() { return 0; }
double *g_ptr;

// CHECK-LABEL: define {{.*void}} @{{.*}}simple{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void simple(float *a, float *b, float *c, float *d) {
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: [[K0:%.+]] = call {{.*}}i64 @{{.*}}get_val
// CHECK-NEXT: store i64 [[K0]], i64* [[K_VAR:%[^,]+]]
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: store i32 12, i32* [[LIN_VAR:%[^,]+]]
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: store i32 -1, i32* [[A:%.+]],
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CHECK: store i32 -1, i32* [[R:%[^,]+]],
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
  #pragma omp parallel for simd
// CHECK: call void @__kmpc_for_static_init_4(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i32* [[LB:%[^,]+]], i32* [[UB:%[^,]+]], i32* [[STRIDE:%[^,]+]], i32 1, i32 1)
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i32 [[UB_VAL]], 5
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i32 [ 5, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i32 [[UP]], i32* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[OMP_IV:%[^,]+]],

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP]], label %[[SIMPLE_LOOP1_BODY:.+]], label %[[SIMPLE_LOOP1_END:[^,]+]]
  for (int i = 3; i < 32; i += 5) {
// CHECK: [[SIMPLE_LOOP1_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 5
// CHECK-NEXT: [[CALC_I_2:%.+]] = add nsw i32 3, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float*
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// br label %{{.+}}, !llvm.loop !{{.+}}
  }
// CHECK: [[SIMPLE_LOOP1_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})

  long long k = get_val();

  #pragma omp parallel for simd linear(k : 3) schedule(dynamic)
// CHECK: [[K0LOAD:%.+]] = load i64, i64* [[K_VAR:%[^,]+]]
// CHECK-NEXT: store i64 [[K0LOAD]], i64* [[LIN0:%[^,]+]]

// CHECK: call void @__kmpc_dispatch_init_4(%ident_t* {{.+}}, i32 %{{.+}}, i32 35, i32 0, i32 8, i32 1, i32 1)
// CHECK: [[NEXT:%.+]] = call i32 @__kmpc_dispatch_next_4(%ident_t* {{.+}}, i32 %{{.+}}, i32* %{{.+}}, i32* [[LB:%.+]], i32* [[UB:%.+]], i32* %{{.+}})
// CHECK: [[COND:%.+]] = icmp ne i32 [[NEXT]], 0
// CHECK: br i1 [[COND]], label %[[CONT:.+]], label %[[END:.+]]
// CHECK: [[CONT]]:
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[OMP_IV2:%[^,]+]],

// CHECK: [[IV2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID:[0-9]+]]
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// CHECK-NEXT: [[CMP2:%.+]] = icmp sle i32 [[IV2]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP2]], label %[[SIMPLE_LOOP2_BODY:.+]], label %[[SIMPLE_LOOP2_END:[^,]+]]
  for (int i = 10; i > 1; i--) {
// CHECK: [[SIMPLE_LOOP2_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV2_0:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// FIXME: It is interesting, why the following "mul 1" was not constant folded?
// CHECK-NEXT: [[IV2_1:%.+]] = mul nsw i32 [[IV2_0]], 1
// CHECK-NEXT: [[LC_I_1:%.+]] = sub nsw i32 10, [[IV2_1]]
// CHECK-NEXT: store i32 [[LC_I_1]], i32* {{.+}}, !llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
//
// CHECK-NEXT: [[LIN0_1:%.+]] = load i64, i64* [[LIN0]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// CHECK-NEXT: [[IV2_2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// CHECK-NEXT: [[LIN_MUL1:%.+]] = mul nsw i32 [[IV2_2]], 3
// CHECK-NEXT: [[LIN_EXT1:%.+]] = sext i32 [[LIN_MUL1]] to i64
// CHECK-NEXT: [[LIN_ADD1:%.+]] = add nsw i64 [[LIN0_1]], [[LIN_EXT1]]
// Update of the privatized version of linear variable!
// CHECK-NEXT: store i64 [[LIN_ADD1]], i64* [[K_PRIVATIZED:%[^,]+]]
    a[k]++;
    k = k + 3;
// CHECK: [[IV2_2:%.+]] = load i32, i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// CHECK-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV2_2]], 1
// CHECK-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV2]]{{.*}}!llvm.mem.parallel_loop_access ![[SIMPLE_LOOP2_ID]]
// br label {{.+}}, !llvm.loop ![[SIMPLE_LOOP2_ID]]
  }
// CHECK: [[SIMPLE_LOOP2_END]]:
//
// Update linear vars after loop, as the loop was operating on a private version.
// CHECK: [[LIN0_2:%.+]] = load i64, i64* [[LIN0]]
// CHECK-NEXT: [[LIN_ADD2:%.+]] = add nsw i64 [[LIN0_2]], 27
// CHECK-NEXT: store i64 [[LIN_ADD2]], i64* %{{.+}}

  int lin = 12;
  #pragma omp parallel for simd linear(lin : get_val()), linear(g_ptr)

// Init linear private var.
// CHECK: [[LIN_VAR:%.+]] = load i32*, i32** %
// CHECK: [[LIN_LOAD:%.+]] = load i32, i32* [[LIN_VAR]]
// CHECK-NEXT: store i32 [[LIN_LOAD]], i32* [[LIN_START:%[^,]+]]
// Remember linear step.
// CHECK: [[CALL_VAL:%.+]] = invoke
// CHECK: store i64 [[CALL_VAL]], i64* [[LIN_STEP:%[^,]+]]

// CHECK: [[GLIN_LOAD:%.+]] = load double*, double** [[GLIN_VAR:%.+]],
// CHECK-NEXT: store double* [[GLIN_LOAD]], double** [[GLIN_START:%[^,]+]]

// CHECK: call void @__kmpc_for_static_init_8u(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i64* [[LB:%[^,]+]], i64* [[UB:%[^,]+]], i64* [[STRIDE:%[^,]+]], i64 1, i64 1)
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ugt i64 [[UB_VAL]], 3
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i64 [ 3, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i64 [[UP]], i64* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[OMP_IV3:%[^,]+]],

// CHECK: [[IV3:%.+]] = load i64, i64* [[OMP_IV3]]
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]]
// CHECK-NEXT: [[CMP3:%.+]] = icmp ule i64 [[IV3]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP3]], label %[[SIMPLE_LOOP3_BODY:.+]], label %[[SIMPLE_LOOP3_END:[^,]+]]
  for (unsigned long long it = 2000; it >= 600; it-=400) {
// CHECK: [[SIMPLE_LOOP3_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV3_0:%.+]] = load i64, i64* [[OMP_IV3]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul i64 [[IV3_0]], 400
// CHECK-NEXT: [[LC_IT_2:%.+]] = sub i64 2000, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* {{.+}}
//
// Linear start and step are used to calculate current value of the linear variable.
// CHECK: [[LINSTART:.+]] = load i32, i32* [[LIN_START]]
// CHECK: [[LINSTEP:.+]] = load i64, i64* [[LIN_STEP]]
// CHECK-NOT: store i32 {{.+}}, i32* [[LIN_VAR]]
// CHECK: [[GLINSTART:.+]] = load double*, double** [[GLIN_START]]
// CHECK-NEXT: [[IV3_1:%.+]] = load i64, i64* [[OMP_IV3]]
// CHECK-NEXT: [[MUL:%.+]] = mul i64 [[IV3_1]], 1
// CHECK: [[GEP:%.+]] = getelementptr{{.*}}[[GLINSTART]]
// CHECK-NEXT: store double* [[GEP]], double** [[G_PTR_CUR:%[^,]+]]
    *g_ptr++ = 0.0;
// CHECK: [[GEP_VAL:%.+]] = load double{{.*}}[[G_PTR_CUR]]
// CHECK: store double{{.*}}[[GEP_VAL]]
    a[it + lin]++;
// CHECK: [[FLT_INC:%.+]] = fadd float
// CHECK-NEXT: store float [[FLT_INC]],
// CHECK: [[IV3_2:%.+]] = load i64, i64* [[OMP_IV3]]
// CHECK-NEXT: [[ADD3_2:%.+]] = add i64 [[IV3_2]], 1
// CHECK-NEXT: store i64 [[ADD3_2]], i64* [[OMP_IV3]]
  }
// CHECK: [[SIMPLE_LOOP3_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
//
// Linear start and step are used to calculate final value of the linear variables.
// CHECK: [[LINSTART:.+]] = load i32, i32* [[LIN_START]]
// CHECK: [[LINSTEP:.+]] = load i64, i64* [[LIN_STEP]]
// CHECK: store i32 {{.+}}, i32* [[LIN_VAR]],
// CHECK: [[GLINSTART:.+]] = load double*, double** [[GLIN_START]]
// CHECK: store double* {{.*}}[[GLIN_VAR]]

  #pragma omp parallel for simd
// CHECK: call void @__kmpc_for_static_init_4(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i32* [[LB:%[^,]+]], i32* [[UB:%[^,]+]], i32* [[STRIDE:%[^,]+]], i32 1, i32 1)
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i32 [[UB_VAL]], 3
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i32 [ 3, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i32 [[UP]], i32* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[OMP_IV4:%[^,]+]],

// CHECK: [[IV4:%.+]] = load i32, i32* [[OMP_IV4]]
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]]
// CHECK-NEXT: [[CMP4:%.+]] = icmp sle i32 [[IV4]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP4]], label %[[SIMPLE_LOOP4_BODY:.+]], label %[[SIMPLE_LOOP4_END:[^,]+]]
  for (short it = 6; it <= 20; it-=-4) {
// CHECK: [[SIMPLE_LOOP4_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV4_0:%.+]] = load i32, i32* [[OMP_IV4]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i32 [[IV4_0]], 4
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i32 6, [[LC_IT_1]]
// CHECK-NEXT: [[LC_IT_3:%.+]] = trunc i32 [[LC_IT_2]] to i16
// CHECK-NEXT: store i16 [[LC_IT_3]], i16*

// CHECK: [[IV4_2:%.+]] = load i32, i32* [[OMP_IV4]]
// CHECK-NEXT: [[ADD4_2:%.+]] = add nsw i32 [[IV4_2]], 1
// CHECK-NEXT: store i32 [[ADD4_2]], i32* [[OMP_IV4]]
  }
// CHECK: [[SIMPLE_LOOP4_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})

  #pragma omp parallel for simd
// CHECK: call void @__kmpc_for_static_init_4(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i32* [[LB:%[^,]+]], i32* [[UB:%[^,]+]], i32* [[STRIDE:%[^,]+]], i32 1, i32 1)
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i32 [[UB_VAL]], 25
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i32 [ 25, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i32 [[UP]], i32* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[OMP_IV5:%[^,]+]],

// CHECK: [[IV5:%.+]] = load i32, i32* [[OMP_IV5]]
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]]
// CHECK-NEXT: [[CMP5:%.+]] = icmp sle i32 [[IV5]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP5]], label %[[SIMPLE_LOOP5_BODY:.+]], label %[[SIMPLE_LOOP5_END:[^,]+]]
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
// CHECK: [[SIMPLE_LOOP5_BODY]]:
// Start of body: calculate it from IV:
// CHECK: [[IV5_0:%.+]] = load i32, i32* [[OMP_IV5]]
// CHECK-NEXT: [[IV5_1:%.+]] = mul nsw i32 [[IV5_0]], 1
// CHECK-NEXT: [[LC_IT_1:%.+]] = sub nsw i32 122, [[IV5_1]]
// CHECK-NEXT: [[LC_IT_2:%.+]] = trunc i32 [[LC_IT_1]] to i8
// CHECK-NEXT: store i8 [[LC_IT_2]], i8* {{.+}},

// CHECK: [[IV5_2:%.+]] = load i32, i32* [[OMP_IV5]]
// CHECK-NEXT: [[ADD5_2:%.+]] = add nsw i32 [[IV5_2]], 1
// CHECK-NEXT: store i32 [[ADD5_2]], i32* [[OMP_IV5]]
  }
// CHECK: [[SIMPLE_LOOP5_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})

// CHECK-NOT: mul i32 %{{.+}}, 10
  #pragma omp parallel for simd
  for (unsigned i=100; i<10; i+=10) {
  }

  int A;
  {
  A = -1;
  #pragma omp parallel for simd lastprivate(A)
// CHECK: call void @__kmpc_for_static_init_8(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i64* [[LB:%[^,]+]], i64* [[UB:%[^,]+]], i64* [[STRIDE:%[^,]+]], i64 1, i64 1)
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i64 [[UB_VAL]], 6
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i64 [ 6, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i64 [[UP]], i64* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[OMP_IV7:%[^,]+]],

// CHECK: br label %[[SIMD_LOOP7_COND:[^,]+]]
// CHECK: [[SIMD_LOOP7_COND]]:
// CHECK-NEXT: [[IV7:%.+]] = load i64, i64* [[OMP_IV7]]
// CHECK-NEXT: [[UB_VAL:%.+]] = load i64, i64* [[UB]]
// CHECK-NEXT: [[CMP7:%.+]] = icmp sle i64 [[IV7]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP7]], label %[[SIMPLE_LOOP7_BODY:.+]], label %[[SIMPLE_LOOP7_END:[^,]+]]
  for (long long i = -10; i < 10; i += 3) {
// CHECK: [[SIMPLE_LOOP7_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV7_0:%.+]] = load i64, i64* [[OMP_IV7]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV7_0]], 3
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* [[LC:%[^,]+]],
// CHECK-NEXT: [[LC_VAL:%.+]] = load i64, i64* [[LC]]
// CHECK-NEXT: [[CONV:%.+]] = trunc i64 [[LC_VAL]] to i32
// CHECK-NEXT: store i32 [[CONV]], i32* [[A_PRIV:%[^,]+]],
    A = i;
// CHECK: [[IV7_2:%.+]] = load i64, i64* [[OMP_IV7]]
// CHECK-NEXT: [[ADD7_2:%.+]] = add nsw i64 [[IV7_2]], 1
// CHECK-NEXT: store i64 [[ADD7_2]], i64* [[OMP_IV7]]
  }
// CHECK: [[SIMPLE_LOOP7_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
// CHECK: load i32, i32*
// CHECK: icmp ne i32 %{{.+}}, 0
// CHECK: br i1 %{{.+}}, label
// CHECK: [[A_PRIV_VAL:%.+]] = load i32, i32* [[A_PRIV]],
// CHECK-NEXT: store i32 [[A_PRIV_VAL]], i32* %{{.+}},
// CHECK-NEXT: br label
  }
  int R;
  {
  R = -1;
// CHECK: store i32 1, i32* [[R_PRIV:%[^,]+]],
  #pragma omp parallel for simd reduction(*:R)
// CHECK: call void @__kmpc_for_static_init_8(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i64* [[LB:%[^,]+]], i64* [[UB:%[^,]+]], i64* [[STRIDE:%[^,]+]], i64 1, i64 1)
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i64 [[UB_VAL]], 6
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i64 [ 6, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i64 [[UP]], i64* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[OMP_IV8:%[^,]+]],

// CHECK: br label %[[SIMD_LOOP8_COND:[^,]+]]
// CHECK: [[SIMD_LOOP8_COND]]:
// CHECK-NEXT: [[IV8:%.+]] = load i64, i64* [[OMP_IV8]]
// CHECK-NEXT: [[UB_VAL:%.+]] = load i64, i64* [[UB]]
// CHECK-NEXT: [[CMP8:%.+]] = icmp sle i64 [[IV8]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP8]], label %[[SIMPLE_LOOP8_BODY:.+]], label %[[SIMPLE_LOOP8_END:[^,]+]]
  for (long long i = -10; i < 10; i += 3) {
// CHECK: [[SIMPLE_LOOP8_BODY]]:
// Start of body: calculate i from IV:
// CHECK: [[IV8_0:%.+]] = load i64, i64* [[OMP_IV8]]
// CHECK-NEXT: [[LC_IT_1:%.+]] = mul nsw i64 [[IV8_0]], 3
// CHECK-NEXT: [[LC_IT_2:%.+]] = add nsw i64 -10, [[LC_IT_1]]
// CHECK-NEXT: store i64 [[LC_IT_2]], i64* [[LC:%[^,]+]],
// CHECK-NEXT: [[LC_VAL:%.+]] = load i64, i64* [[LC]]
// CHECK: store i32 %{{.+}}, i32* [[R_PRIV]],
    R *= i;
// CHECK: [[IV8_2:%.+]] = load i64, i64* [[OMP_IV8]]
// CHECK-NEXT: [[ADD8_2:%.+]] = add nsw i64 [[IV8_2]], 1
// CHECK-NEXT: store i64 [[ADD8_2]], i64* [[OMP_IV8]]
  }
// CHECK: [[SIMPLE_LOOP8_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
// CHECK: call i32 @__kmpc_reduce_nowait(
// CHECK: [[R_PRIV_VAL:%.+]] = load i32, i32* [[R_PRIV]],
// CHECK: [[RED:%.+]] = mul nsw i32 %{{.+}}, [[R_PRIV_VAL]]
// CHECK-NEXT: store i32 [[RED]], i32* %{{.+}},
// CHECK-NEXT: call void @__kmpc_end_reduce_nowait(
  }
}

template <class T, unsigned K> T tfoo(T a) { return a + K; }

template <typename T, unsigned N>
int templ1(T a, T *z) {
  #pragma omp parallel for simd collapse(N)
  for (int i = 0; i < N * 2; i++) {
    for (long long j = 0; j < (N + N + N + N); j += 2) {
      z[i + j] = a + tfoo<T, N>(i + j);
    }
  }
  return 0;
}

// Instatiation templ1<float,2>
// CHECK-LABEL: define {{.*i32}} @{{.*}}templ1{{.*}}(float {{.+}}, float* {{.+}})
// CHECK: call void (%ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
void inst_templ1() {
  float a;
  float z[100];
  templ1<float,2> (a, z);
}


typedef int MyIdx;

class IterDouble {
  double *Ptr;
public:
  IterDouble operator++ () const {
    IterDouble n;
    n.Ptr = Ptr + 1;
    return n;
  }
  bool operator < (const IterDouble &that) const {
    return Ptr < that.Ptr;
  }
  double & operator *() const {
    return *Ptr;
  }
  MyIdx operator - (const IterDouble &that) const {
    return (MyIdx) (Ptr - that.Ptr);
  }
  IterDouble operator + (int Delta) {
    IterDouble re;
    re.Ptr = Ptr + Delta;
    return re;
  }

  ///~IterDouble() {}
};

// CHECK-LABEL: define {{.*void}} @{{.*}}iter_simple{{.*}}
void iter_simple(IterDouble ia, IterDouble ib, IterDouble ic) {
//
// Calculate number of iterations before the loop body.
// CHECK: [[DIFF1:%.+]] = invoke {{.*}}i32 @{{.*}}IterDouble{{.*}}
// CHECK: [[DIFF2:%.+]] = sub nsw i32 [[DIFF1]], 1
// CHECK-NEXT: [[DIFF3:%.+]] = add nsw i32 [[DIFF2]], 1
// CHECK-NEXT: [[DIFF4:%.+]] = sdiv i32 [[DIFF3]], 1
// CHECK-NEXT: [[DIFF5:%.+]] = sub nsw i32 [[DIFF4]], 1
// CHECK-NEXT: store i32 [[DIFF5]], i32* [[OMP_LAST_IT:%[^,]+]]{{.+}}
  #pragma omp parallel for simd

// CHECK: call void @__kmpc_for_static_init_4(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i32* [[LB:%[^,]+]], i32* [[UB:%[^,]+]], i32* [[STRIDE:%[^,]+]], i32 1, i32 1)
// CHECK-DAG: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK-DAG: [[OMP_LAST_IT_VAL:%.+]] = load i32, i32* [[OMP_LAST_IT]],
// CHECK: [[CMP:%.+]] = icmp sgt i32 [[UB_VAL]], [[OMP_LAST_IT_VAL]]
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: [[OMP_LAST_IT_VAL:%.+]] = load i32, i32* [[OMP_LAST_IT]],
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i32 [ [[OMP_LAST_IT_VAL]], %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i32 [[UP]], i32* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[IT_OMP_IV:%[^,]+]],

// CHECK: [[IV:%.+]] = load i32, i32* [[IT_OMP_IV]]
// CHECK-NEXT: [[UB_VAL:%.+]] = load i32, i32* [[UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP]], label %[[IT_BODY:[^,]+]], label %[[IT_END:[^,]+]]
  for (IterDouble i = ia; i < ib; ++i) {
// CHECK: [[IT_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32, i32* [[IT_OMP_IV]]
// Call of operator+ (i, IV).
// CHECK: {{%.+}} = invoke {{.+}} @{{.*}}IterDouble{{.*}}
// ... loop body ...
   *i = *ic * 0.5;
// Float multiply and save result.
// CHECK: [[MULR:%.+]] = fmul double {{%.+}}, 5.000000e-01
// CHECK-NEXT: invoke {{.+}} @{{.*}}IterDouble{{.*}}
// CHECK: store double [[MULR:%.+]], double* [[RESULT_ADDR:%.+]]
   ++ic;
//
// CHECK: [[IV2:%.+]] = load i32, i32* [[IT_OMP_IV]]
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[IT_OMP_IV]]
// br label %{{.*}}, !llvm.loop ![[ITER_LOOP_ID]]
  }
// CHECK: [[IT_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
// CHECK: ret void
}


// CHECK-LABEL: define {{.*void}} @{{.*}}collapsed{{.*}}
void collapsed(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  unsigned j; // middle loop couter, leads to unsigned icmp in loop header.
  // k declared in the loop init below
  short l; // inner loop counter
// CHECK: call void @__kmpc_for_static_init_4u(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i32* [[LB:%[^,]+]], i32* [[UB:%[^,]+]], i32* [[STRIDE:%[^,]+]], i32 1, i32 1)
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: [[CMP:%.+]] = icmp ugt i32 [[UB_VAL]], 119
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i32 [ 119, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i32 [[UP]], i32* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i32, i32* [[LB]],
// CHECK: store i32 [[LB_VAL]], i32* [[OMP_IV:%[^,]+]],
//
  #pragma omp parallel for simd collapse(4)

// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK: [[UB_VAL:%.+]] = load i32, i32* [[UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ule i32 [[IV]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP]], label %[[COLL1_BODY:[^,]+]], label %[[COLL1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 2u; j < 5u; j++) //3 iterations
      for (int k = 3; k <= 6; k++) // 4 iterations
        for (l = 4; l < 9; ++l) // 5 iterations
        {
// CHECK: [[COLL1_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i32, i32* [[OMP_IV]]
// Calculation of the loop counters values.
// CHECK: [[CALC_I_1:%.+]] = udiv i32 [[IV1]], 60
// CHECK-NEXT: [[CALC_I_1_MUL1:%.+]] = mul i32 [[CALC_I_1]], 1
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i32 1, [[CALC_I_1_MUL1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_J_1:%.+]] = udiv i32 [[IV1_2]], 20
// CHECK-NEXT: [[CALC_J_2:%.+]] = urem i32 [[CALC_J_1]], 3
// CHECK-NEXT: [[CALC_J_2_MUL1:%.+]] = mul i32 [[CALC_J_2]], 1
// CHECK-NEXT: [[CALC_J_3:%.+]] = add i32 2, [[CALC_J_2_MUL1]]
// CHECK-NEXT: store i32 [[CALC_J_3]], i32* [[LC_J:.+]]
// CHECK: [[IV1_3:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_K_1:%.+]] = udiv i32 [[IV1_3]], 5
// CHECK-NEXT: [[CALC_K_2:%.+]] = urem i32 [[CALC_K_1]], 4
// CHECK-NEXT: [[CALC_K_2_MUL1:%.+]] = mul i32 [[CALC_K_2]], 1
// CHECK-NEXT: [[CALC_K_3:%.+]] = add i32 3, [[CALC_K_2_MUL1]]
// CHECK-NEXT: store i32 [[CALC_K_3]], i32* [[LC_K:.+]]
// CHECK: [[IV1_4:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_L_1:%.+]] = urem i32 [[IV1_4]], 5
// CHECK-NEXT: [[CALC_L_1_MUL1:%.+]] = mul i32 [[CALC_L_1]], 1
// CHECK-NEXT: [[CALC_L_2:%.+]] = add i32 4, [[CALC_L_1_MUL1]]
// CHECK-NEXT: [[CALC_L_3:%.+]] = trunc i32 [[CALC_L_2]] to i16
// CHECK-NEXT: store i16 [[CALC_L_3]], i16* [[LC_L:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]
    float res = b[j] * c[k];
    a[i] = res * d[l];
// CHECK: [[IV2:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[ADD2:%.+]] = add i32 [[IV2]], 1
// CHECK-NEXT: store i32 [[ADD2]], i32* [[OMP_IV]]
// br label %{{[^,]+}}, !llvm.loop ![[COLL1_LOOP_ID]]
// CHECK: [[COLL1_END]]:
  }
// i,j,l are updated; k is not updated.
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
// CHECK: store i32 3, i32* [[I:%[^,]+]]
// CHECK: store i32 5, i32* [[I:%[^,]+]]
// CHECK: store i16 9, i16* [[I:%[^,]+]]
// CHECK: ret void
}

extern char foo();
extern double globalfloat;

// CHECK-LABEL: define {{.*void}} @{{.*}}widened{{.*}}
void widened(float *a, float *b, float *c, float *d) {
  int i; // outer loop counter
  short j; // inner loop counter
  globalfloat = 1.0;
  int localint = 1;
// CHECK: store double {{.+}}, double* [[GLOBALFLOAT:@.+]]
// Counter is widened to 64 bits.
// CHECK:     [[MUL:%.+]] = mul nsw i64 2, %{{.+}}
// CHECK-NEXT: [[SUB:%.+]] = sub nsw i64 [[MUL]], 1
// CHECK-NEXT: store i64 [[SUB]], i64* [[OMP_LAST_IT:%[^,]+]],
// CHECK: call void @__kmpc_for_static_init_8(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i64* [[LB:%[^,]+]], i64* [[UB:%[^,]+]], i64* [[STRIDE:%[^,]+]], i64 1, i64 1)
// CHECK-DAG: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK-DAG: [[OMP_LAST_IT_VAL:%.+]] = load i64, i64* [[OMP_LAST_IT]],
// CHECK: [[CMP:%.+]] = icmp sgt i64 [[UB_VAL]], [[OMP_LAST_IT_VAL]]
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: [[OMP_LAST_IT_VAL:%.+]] = load i64, i64* [[OMP_LAST_IT]],
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i64 [ [[OMP_LAST_IT_VAL]], %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i64 [[UP]], i64* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[OMP_IV:%[^,]+]],
//
  #pragma omp parallel for simd collapse(2) private(globalfloat, localint)

// CHECK: [[IV:%.+]] = load i64, i64* [[OMP_IV]]
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i64 [[IV]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP]], label %[[WIDE1_BODY:[^,]+]], label %[[WIDE1_END:[^,]+]]
  for (i = 1; i < 3; i++) // 2 iterations
    for (j = 0; j < foo(); j++) // foo() iterations
  {
// CHECK: [[WIDE1_BODY]]:
// Start of body: calculate i from index:
// CHECK: [[IV1:%.+]] = load i64, i64* [[OMP_IV]]
// Calculation of the loop counters values...
// CHECK: store i32 {{[^,]+}}, i32* [[LC_I:.+]]
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]
// CHECK: store i16 {{[^,]+}}, i16* [[LC_J:.+]]
// ... loop body ...
//
// Here we expect store into private double var, not global
// CHECK-NOT: store double {{.+}}, double* [[GLOBALFLOAT]]
    globalfloat = (float)j/i;
    float res = b[j] * c[j];
// Store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* [[RESULT_ADDR:%.+]]
    a[i] = res * d[i];
// Then there's a store into private var localint:
// CHECK: store i32 {{.+}}, i32* [[LOCALINT:%[^,]+]]
    localint = (int)j;
// CHECK: [[IV2:%.+]] = load i64, i64* [[OMP_IV]]
// CHECK-NEXT: [[ADD2:%.+]] = add nsw i64 [[IV2]], 1
// CHECK-NEXT: store i64 [[ADD2]], i64* [[OMP_IV]]
//
// br label %{{[^,]+}}, !llvm.loop ![[WIDE1_LOOP_ID]]
// CHECK: [[WIDE1_END]]:
  }
// i,j are updated.
// CHECK: store i32 3, i32* [[I:%[^,]+]]
// CHECK: store i16
//
// Here we expect store into original localint, not its privatized version.
// CHECK-NOT: store i32 {{.+}}, i32* [[LOCALINT]]
  localint = (int)j;
// CHECK: ret void
}

// CHECK: call void @__kmpc_for_static_init_8(%ident_t* {{[^,]+}}, i32 %{{[^,]+}}, i32 34, i32* %{{[^,]+}}, i64* [[LB:%[^,]+]], i64* [[UB:%[^,]+]], i64* [[STRIDE:%[^,]+]], i64 1, i64 1)
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: [[CMP:%.+]] = icmp sgt i64 [[UB_VAL]], 15
// CHECK: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:[^,]+]]
// CHECK: [[TRUE]]:
// CHECK: br label %[[SWITCH:[^,]+]]
// CHECK: [[FALSE]]:
// CHECK: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// CHECK: br label %[[SWITCH]]
// CHECK: [[SWITCH]]:
// CHECK: [[UP:%.+]] = phi i64 [ 15, %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// CHECK: store i64 [[UP]], i64* [[UB]],
// CHECK: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// CHECK: store i64 [[LB_VAL]], i64* [[T1_OMP_IV:%[^,]+]],

// ...
// CHECK: [[IV:%.+]] = load i64, i64* [[T1_OMP_IV]]
// CHECK-NEXT: [[UB_VAL:%.+]] = load i64, i64* [[UB]]
// CHECK-NEXT: [[CMP1:%.+]] = icmp sle i64 [[IV]], [[UB_VAL]]
// CHECK-NEXT: br i1 [[CMP1]], label %[[T1_BODY:.+]], label %[[T1_END:[^,]+]]
// CHECK: [[T1_BODY]]:
// Loop counters i and j updates:
// CHECK: [[IV1:%.+]] = load i64, i64* [[T1_OMP_IV]]
// CHECK-NEXT: [[I_1:%.+]] = sdiv i64 [[IV1]], 4
// CHECK-NEXT: [[I_1_MUL1:%.+]] = mul nsw i64 [[I_1]], 1
// CHECK-NEXT: [[I_1_ADD0:%.+]] = add nsw i64 0, [[I_1_MUL1]]
// CHECK-NEXT: [[I_2:%.+]] = trunc i64 [[I_1_ADD0]] to i32
// CHECK-NEXT: store i32 [[I_2]], i32*
// CHECK: [[IV2:%.+]] = load i64, i64* [[T1_OMP_IV]]
// CHECK-NEXT: [[J_1:%.+]] = srem i64 [[IV2]], 4
// CHECK-NEXT: [[J_2:%.+]] = mul nsw i64 [[J_1]], 2
// CHECK-NEXT: [[J_2_ADD0:%.+]] = add nsw i64 0, [[J_2]]
// CHECK-NEXT: store i64 [[J_2_ADD0]], i64*
// simd.for.inc:
// CHECK: [[IV3:%.+]] = load i64, i64* [[T1_OMP_IV]]
// CHECK-NEXT: [[INC:%.+]] = add nsw i64 [[IV3]], 1
// CHECK-NEXT: store i64 [[INC]], i64*
// CHECK-NEXT: br label {{%.+}}
// CHECK: [[T1_END]]:
// CHECK: call void @__kmpc_for_static_fini(%ident_t* {{.+}}, i32 %{{.+}})
// CHECK: ret void
//
// TERM_DEBUG-LABEL: bar
int bar() {return 0;};

// TERM_DEBUG-LABEL: parallel_simd
void parallel_simd(float *a) {
#pragma omp parallel for simd
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     invoke i32 {{.*}}bar{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += bar();
}
// TERM_DEBUG: !{{[0-9]+}} = !DILocation(line: [[@LINE-11]],
#endif // HEADER

