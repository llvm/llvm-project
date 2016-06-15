// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
//
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER
// CHECK: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-DAG: [[I:@.+]] = global i8 1,
// CHECK-DAG: [[J:@.+]] = global i8 2,
// CHECK-DAG: [[K:@.+]] = global i8 3,

// CHECK-LABEL: define {{.*void}} @{{.*}}without_schedule_clause{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void without_schedule_clause(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for nowait
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// UB = min(UB, GlobalUB)
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4571423
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 4571423, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// Loop header
// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (int i = 33; i < 32000000; i += 7) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 7
// CHECK-NEXT: [[CALC_I_2:%.+]] = add nsw i32 33, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NOT: __kmpc_barrier
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_not_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_not_chunked(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(static)
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// UB = min(UB, GlobalUB)
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4571423
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 4571423, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// Loop header
// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (int i = 32000000; i > 33; i += -7) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 7
// CHECK-NEXT: [[CALC_I_2:%.+]] = sub nsw i32 32000000, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_chunked(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(monotonic: static, 5)
// CHECK: call void @__kmpc_for_static_init_4u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 536870945, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 5)
// UB = min(UB, GlobalUB)
// CHECK: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[UBCMP:%.+]] = icmp ugt i32 [[UB]], 16908288
// CHECK-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// CHECK: [[UBRESULT:%.+]] = phi i32 [ 16908288, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// CHECK-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// CHECK-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]

// Outer loop header
// CHECK: [[O_IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[O_UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ule i32 [[O_IV]], [[O_UB]]
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ule i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (unsigned i = 131071; i <= 2147483647; i += 127) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul i32 [[IV1_1]], 127
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i32 131071, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// Update the counters, adding stride
// CHECK:  [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: [[ST:%.+]] = load i32, i32* [[OMP_ST]]
// CHECK-NEXT: [[ADD_LB:%.+]] = add i32 [[LB]], [[ST]]
// CHECK-NEXT: store i32 [[ADD_LB]], i32* [[OMP_LB]]
// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[ST:%.+]] = load i32, i32* [[OMP_ST]]
// CHECK-NEXT: [[ADD_UB:%.+]] = add i32 [[UB]], [[ST]]
// CHECK-NEXT: store i32 [[ADD_UB]], i32* [[OMP_UB]]

// CHECK: [[O_LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}dynamic1{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void dynamic1(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(nonmonotonic: dynamic)
// CHECK: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 1073741859, i64 0, i64 16908287, i64 1, i64 1)
//
// CHECK: [[HASWORK:%.+]] = call i32 @__kmpc_dispatch_next_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32* [[OMP_ISLAST:%[^,]+]], i64* [[OMP_LB:%[^,]+]], i64* [[OMP_UB:%[^,]+]], i64* [[OMP_ST:%[^,]+]])
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ne i32 [[HASWORK]], 0
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[LB:%.+]] = load i64, i64* [[OMP_LB]]
// CHECK-NEXT: store i64 [[LB]], i64* [[OMP_IV:[^,]+]]
// CHECK: [[IV:%.+]] = load i64, i64* [[OMP_IV]]

// CHECK-NEXT: [[UB:%.+]] = load i64, i64* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ule i64 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (unsigned long long i = 131071; i < 2147483647; i += 127) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i64, i64* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul i64 [[IV1_1]], 127
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i64 131071, [[CALC_I_1]]
// CHECK-NEXT: store i64 [[CALC_I_2]], i64* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}!llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}guided7{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void guided7(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(guided, 7)
// CHECK: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 36, i64 0, i64 16908287, i64 1, i64 7)
//
// CHECK: [[HASWORK:%.+]] = call i32 @__kmpc_dispatch_next_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32* [[OMP_ISLAST:%[^,]+]], i64* [[OMP_LB:%[^,]+]], i64* [[OMP_UB:%[^,]+]], i64* [[OMP_ST:%[^,]+]])
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ne i32 [[HASWORK]], 0
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[LB:%.+]] = load i64, i64* [[OMP_LB]]
// CHECK-NEXT: store i64 [[LB]], i64* [[OMP_IV:[^,]+]]
// CHECK: [[IV:%.+]] = load i64, i64* [[OMP_IV]]

// CHECK-NEXT: [[UB:%.+]] = load i64, i64* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp ule i64 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (unsigned long long i = 131071; i < 2147483647; i += 127) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i64, i64* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul i64 [[IV1_1]], 127
// CHECK-NEXT: [[CALC_I_2:%.+]] = add i64 131071, [[CALC_I_1]]
// CHECK-NEXT: store i64 [[CALC_I_2]], i64* [[LC_I:.+]]
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}!llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}test_auto{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void test_auto(float *a, float *b, float *c, float *d) {
  unsigned int x = 0;
  unsigned int y = 0;
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(auto) collapse(2)
// CHECK: call void @__kmpc_dispatch_init_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 38, i64 0, i64 [[LAST_ITER:%[^,]+]], i64 1, i64 1)
//
// CHECK: [[HASWORK:%.+]] = call i32 @__kmpc_dispatch_next_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32* [[OMP_ISLAST:%[^,]+]], i64* [[OMP_LB:%[^,]+]], i64* [[OMP_UB:%[^,]+]], i64* [[OMP_ST:%[^,]+]])
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ne i32 [[HASWORK]], 0
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[LB:%.+]] = load i64, i64* [[OMP_LB]]
// CHECK-NEXT: store i64 [[LB]], i64* [[OMP_IV:[^,]+]]
// CHECK: [[IV:%.+]] = load i64, i64* [[OMP_IV]]

// CHECK-NEXT: [[UB:%.+]] = load i64, i64* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i64 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
// FIXME: When the iteration count of some nested loop is not a known constant,
// we should pre-calculate it, like we do for the total number of iterations!
  for (char i = static_cast<char>(y); i <= '9'; ++i)
    for (x = 11; x > 0; --x) {
// CHECK: [[LOOP1_BODY]]
// Start of body: indices are calculated from IV:
// CHECK: store i8 {{%[^,]+}}, i8* {{%[^,]+}}
// CHECK: store i32 {{%[^,]+}}, i32* {{%[^,]+}}
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}runtime{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void runtime(float *a, float *b, float *c, float *d) {
  int x = 0;
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for collapse(2) schedule(runtime)
// CHECK: call void @__kmpc_dispatch_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 37, i32 0, i32 199, i32 1, i32 1)
//
// CHECK: [[HASWORK:%.+]] = call i32 @__kmpc_dispatch_next_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32* [[OMP_ISLAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]])
// CHECK-NEXT: [[O_CMP:%.+]] = icmp ne i32 [[HASWORK]], 0
// CHECK-NEXT: br i1 [[O_CMP]], label %[[O_LOOP1_BODY:[^,]+]], label %[[O_LOOP1_END:[^,]+]]

// Loop header
// CHECK: [[O_LOOP1_BODY]]
// CHECK: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// CHECK-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// CHECK: [[IV:%.+]] = load i32, i32* [[OMP_IV]]

// CHECK-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// CHECK-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (unsigned char i = '0' ; i <= '9'; ++i)
    for (x = -10; x < 10; ++x) {
// CHECK: [[LOOP1_BODY]]
// Start of body: indices are calculated from IV:
// CHECK: store i8 {{%[^,]+}}, i8* {{%[^,]+}}
// CHECK: store i32 {{%[^,]+}}, i32* {{%[^,]+}}
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.mem.parallel_loop_access
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: test_precond
void test_precond() {
  // CHECK: [[A_ADDR:%.+]] = alloca i8,
  // CHECK: [[CAP:%.+]] = alloca i8,
  // CHECK: [[I_ADDR:%.+]] = alloca i8,
  char a = 0;
  // CHECK: store i8 0,
  // CHECK: store i32
  // CHECK: store i8
  // CHECK: [[A:%.+]] = load i8, i8* [[CAP]],
  // CHECK: [[CONV:%.+]] = sext i8 [[A]] to i32
  // CHECK: [[CMP:%.+]] = icmp slt i32 [[CONV]], 10
  // CHECK: br i1 [[CMP]], label %[[PRECOND_THEN:[^,]+]], label %[[PRECOND_END:[^,]+]]
  // CHECK: [[PRECOND_THEN]]
  // CHECK: call void @__kmpc_for_static_init_4
#pragma omp for
  for(char i = a; i < 10; ++i);
  // CHECK: call void @__kmpc_for_static_fini
  // CHECK: [[PRECOND_END]]
}

// TERM_DEBUG-LABEL: foo
int foo() {return 0;};

// TERM_DEBUG-LABEL: parallel_for
void parallel_for(float *a) {
#pragma omp parallel
#pragma omp for schedule(static, 5)
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_for_static_init_4u({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke i32 {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_for_static_fini({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     call {{.+}} @__kmpc_barrier({{.+}}), !dbg [[DBG_LOC_CANCEL:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += foo();
}
// Check source line corresponds to "#pragma omp for schedule(static, 5)" above:
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-15]],
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !DILocation(line: [[@LINE-16]],
// TERM_DEBUG-DAG: [[DBG_LOC_CANCEL]] = !DILocation(line: [[@LINE-17]],

char i = 1, j = 2, k = 3;
// CHECK-LABEL: for_with_global_lcv
void for_with_global_lcv() {
// CHECK: [[I_ADDR:%.+]] = alloca i8,
// CHECK: [[J_ADDR:%.+]] = alloca i8,

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: [[I]]
// CHECK: store i8 %{{.+}}, i8* [[I_ADDR]]
// CHECK-NOT: [[I]]
// CHECK: [[I_VAL:%.+]] = load i8, i8* [[I_ADDR]],
// CHECK-NOT: [[I]]
// CHECK: store i8 [[I_VAL]], i8* [[K]]
// CHECK-NOT: [[I]]
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: call void @__kmpc_barrier(
#pragma omp for
  for (i = 0; i < 2; ++i) {
    k = i;
  }
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: [[J]]
// CHECK: store i8 %{{.+}}, i8* [[J_ADDR]]
// CHECK-NOT: [[J]]
// CHECK: [[J_VAL:%.+]] = load i8, i8* [[J_ADDR]],
// CHECK-NOT: [[J]]
// CHECK: store i8 [[J_VAL]], i8* [[K]]
// CHECK-NOT: [[J]]
// CHECK: call void @__kmpc_for_static_fini(
#pragma omp for collapse(2)
  for (int i = 0; i < 2; ++i)
  for (j = 0; j < 2; ++j) {
    k = i;
    k = j;
  }
  char &cnt = i;
#pragma omp for
  for (cnt = 0; cnt < 2; ++cnt)
    k = cnt;
}

// CHECK-LABEL: for_with_references
void for_with_references() {
// CHECK: [[I:%.+]] = alloca i8,
// CHECK: [[CNT:%.+]] = alloca i8*,
// CHECK: [[CNT_PRIV:%.+]] = alloca i8,
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK-NOT: load i8, i8* [[CNT]],
// CHECK: call void @__kmpc_for_static_fini(
  char i = 0;
  char &cnt = i;
#pragma omp for
  for (cnt = 0; cnt < 2; ++cnt)
    k = cnt;
}

struct Bool {
  Bool(bool b) : b(b) {}
  operator bool() const { return b; }
  const bool b;
};

template <typename T>
struct It {
  It() : p(0) {}
  It(const It &, int = 0) ;
  template <typename U>
  It(U &, int = 0) ;
  It &operator=(const It &);
  It &operator=(It &);
  ~It() {}

  It(T *p) : p(p) {}

  operator T *&() { return p; }
  operator T *() const { return p; }
  T *operator->() const { return p; }

  It &operator++() { ++p; return *this; }
  It &operator--() { --p; return *this; }
  It &operator+=(unsigned n) { p += n; return *this; }
  It &operator-=(unsigned n) { p -= n; return *this; }

  T *p;
};

template <typename T>
It<T> operator+(It<T> a, typename It<T>::difference_type n) { return a.p + n; }

template <typename T>
It<T> operator+(typename It<T>::difference_type n, It<T> a) { return a.p + n; }

template <typename T>
It<T> operator-(It<T> a, typename It<T>::difference_type n) { return a.p - n; }

typedef Bool BoolType;

template <typename T>
BoolType operator<(It<T> a, It<T> b) { return a.p < b.p; }

void loop_with_It(It<char> begin, It<char> end) {
#pragma omp for
  for (It<char> it = begin; it < end; ++it) {
    *it = 0;
  }
}

// CHECK-LABEL: loop_with_It
// CHECK: call i32 @__kmpc_global_thread_num(
// CHECK: call void @__kmpc_for_static_init_8(
// CHECK: call void @__kmpc_for_static_fini(

void loop_with_stmt_expr() {
#pragma omp for
  for (int i = __extension__({float b = 0;b; }); i < __extension__({double c = 1;c; }); i += __extension__({char d = 1; d; }))
    ;
}
// CHECK-LABEL: loop_with_stmt_expr
// CHECK: call i32 @__kmpc_global_thread_num(
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(


// CHECK-LABEL: fint
// CHECK: call {{.*}}i32 {{.*}}ftemplate
// CHECK: ret i32

// CHECK: load i16, i16*
// CHECK: store i16 %
// CHECK: call void {{.+}}@__kmpc_fork_call(
// CHECK: call void @__kmpc_for_static_init_4(
template <typename T>
T ftemplate() {
  short aa = 0;

#pragma omp parallel for schedule(static, aa)
  for (int i = 0; i < 100; i++) {
  }
  return T();
}

int fint(void) { return ftemplate<int>(); }

#endif // HEADER
