// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=OMP50,CHECK
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=OMP45,CHECK

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=OMP50,CHECK

// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=OMP45,CHECK

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -gno-column-info -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -O1 -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=CLEANUP

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -O1 -fopenmp-simd -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -DOMP5 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix=OMP5 %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -DOMP5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -DOMP5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix=OMP5 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -DOMP5 -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -DOMP5 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -DOMP5 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
#ifndef HEADER
#define HEADER

#ifndef OMP5
// CHECK-DAG: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[LOOP_LOC:@.+]] = private unnamed_addr constant [[IDENT_T_TY]] { i32 0, i32 514, i32 0, i32 0, i8*

// CHECK-LABEL: with_var_schedule
void with_var_schedule() {
  double a = 5;
// CHECK: [[CHUNK_SIZE:%.+]] = fptosi double %{{.+}}to i8
// CHECK: store i8 %{{.+}}, i8* [[CHUNK:%.+]],
// CHECK: [[VAL:%.+]] = load i8, i8* [[CHUNK]],
// CHECK: store i8 [[VAL]], i8*
// CHECK: [[CHUNK:%.+]] = load i64, i64* %
// CHECK: call void {{.+}} @__kmpc_fork_call({{.+}}, i64 [[CHUNK]])

// CHECK: [[UNDEF_A:%.+]] = load double, double* undef
// CHECK: fadd double 2.000000e+00, [[UNDEF_A]]
// CHECK: [[CHUNK_VAL:%.+]] = load i8, i8* %
// CHECK: [[CHUNK_SIZE:%.+]] = sext i8 [[CHUNK_VAL]] to i64
// CHECK: call void @__kmpc_for_static_init_8u([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID:%[^,]+]], i32 33, i32* [[IS_LAST:%[^,]+]], i64* [[OMP_LB:%[^,]+]], i64* [[OMP_UB:%[^,]+]], i64* [[OMP_ST:%[^,]+]], i64 1, i64 [[CHUNK_SIZE]])
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID:%.+]])
#pragma omp parallel for schedule(static, char(a)) private(a)
  for (unsigned long long i = 1; i < 2 + a; ++i) {
  }
}

// CHECK-LABEL: define {{.*void}} @{{.*}}without_schedule_clause{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void without_schedule_clause(float *a, float *b, float *c, float *d) {
  #pragma omp parallel for
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID:%.+]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
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
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_not_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_not_chunked(float *a, float *b, float *c, float *d) {
  #pragma omp parallel for schedule(static)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// CHECK: call void @__kmpc_for_static_init_4([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID:%.+]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
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
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}static_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_chunked(float *a, float *b, float *c, float *d) {
  #pragma omp parallel for schedule(static, 5)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// CHECK: call void @__kmpc_for_static_init_4u([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID:%.+]], i32 33, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 5)
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
// CHECK: call void @__kmpc_for_static_fini([[IDENT_T_TY]]* [[LOOP_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}dynamic1{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void dynamic1(float *a, float *b, float *c, float *d) {
  #pragma omp parallel for schedule(dynamic)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// OMP45: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 35, i64 0, i64 16908287, i64 1, i64 1)
// OMP50: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 1073741859, i64 0, i64 16908287, i64 1, i64 1)
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
// CHECK-NEXT: [[BOUND:%.+]] = add i64 [[UB]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp ult i64 [[IV]], [[BOUND]]
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
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}guided7{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void guided7(float *a, float *b, float *c, float *d) {
  #pragma omp parallel for schedule(guided, 7)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// OMP45: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 36, i64 0, i64 16908287, i64 1, i64 7)
// OMP50: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 1073741860, i64 0, i64 16908287, i64 1, i64 7)
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
// CHECK-NEXT: [[BOUND:%.+]] = add i64 [[UB]], 1
// CHECK-NEXT: [[CMP:%.+]] = icmp ult i64 [[IV]], [[BOUND]]
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
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}test_auto{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void test_auto(float *a, float *b, float *c, float *d) {
  unsigned int x = 0;
  unsigned int y = 0;
  #pragma omp parallel for schedule(auto) collapse(2)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, i32* nonnull align 4 dereferenceable(4) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// OMP45: call void @__kmpc_dispatch_init_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 38, i64 0, i64 [[LAST_ITER:%[^,]+]], i64 1, i64 1)
// OMP50: call void @__kmpc_dispatch_init_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 1073741862, i64 0, i64 [[LAST_ITER:%[^,]+]], i64 1, i64 1)
//
// CHECK: [[HASWORK:%.+]] = call i32 @__kmpc_dispatch_next_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32* [[OMP_ISLAST:%[^,]+]], i64* [[OMP_LB:%[^,]+]], i64* [[OMP_UB:%[^,]+]], i64* [[OMP_ST:%[^,]+]])
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
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}runtime{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void runtime(float *a, float *b, float *c, float *d) {
  int x = 0;
  #pragma omp parallel for collapse(2) schedule(runtime)
// CHECK: call void ([[IDENT_T_TY]]*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]], i32 4, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float**, float**, float**, float**)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*),
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}}, float** nonnull align 8 dereferenceable(8) %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
// OMP45: call void @__kmpc_dispatch_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 37, i32 0, i32 199, i32 1, i32 1)
// OMP50: call void @__kmpc_dispatch_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID:%.+]], i32 1073741861, i32 0, i32 199, i32 1, i32 1)
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
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: ret void
}

// TERM_DEBUG-LABEL: foo
int foo() {return 0;};

// TERM_DEBUG-LABEL: parallel_for
// CLEANUP: parallel_for
void parallel_for(float *a, const int n) {
  float arr[n];
#pragma omp parallel for schedule(static, 5) private(arr) default(none) firstprivate(n) shared(a)
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_for_static_init_4u({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke i32 {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_for_static_fini({{.+}}), !dbg [[DBG_LOC_START]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  // CLEANUP-NOT: __kmpc_global_thread_num
  // CLEANUP:     call void @__kmpc_for_static_init_4u({{.+}})
  // CLEANUP:     call void @__kmpc_for_static_fini({{.+}})
  for (unsigned i = 131071; i <= 2147483647; i += 127)
    a[i] += foo() + arr[i] + n;
}
// Check source line corresponds to "#pragma omp parallel for schedule(static, 5)" above:
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-17]],

#else // OMP5
// OMP5-LABEL: increment
int increment () {
// OMP5: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for
// Determine UB = min(UB, GlobalUB)
// OMP5: call void @__kmpc_for_static_init_4(%struct.ident_t* [[LOOP_LOC:[@%].+]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// OMP5-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// OMP5-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4
// OMP5-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// OMP5: [[UBRESULT:%.+]] = phi i32 [ 4, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// OMP5-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// OMP5-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// OMP5-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// OMP5-NEXT: br label %[[LOOP1_HEAD:.+]]

// Loop header
// OMP5: [[LOOP1_HEAD]]
// OMP5: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// OMP5-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// OMP5-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// OMP5-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]

  for (int i = 0 ; i != 5; ++i)
// Start of body: calculate i from IV:
// OMP5: [[LOOP1_BODY]]
// OMP5: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// OMP5-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 1
// OMP5-NEXT: [[CALC_I_2:%.+]] = add nsw i32 0, [[CALC_I_1]]
// OMP5-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]
// OMP5: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// OMP5-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// OMP5-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// OMP5-NEXT: br label %[[LOOP1_HEAD]]
    ;
// OMP5: [[LOOP1_END]]
// OMP5: call void @__kmpc_for_static_fini(%struct.ident_t* [[LOOP_LOC]], i32 [[GTID]])
// OMP5: __kmpc_barrier
  return 0;
// OMP5: ret i32 0
}

// OMP5-LABEL: decrement_nowait
int decrement_nowait () {
// OMP5: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for nowait
// Determine UB = min(UB, GlobalUB)
// OMP5: call void @__kmpc_for_static_init_4(%struct.ident_t* [[LOOP_LOC]], i32 [[GTID]], i32 34, i32* [[IS_LAST:%[^,]+]], i32* [[OMP_LB:%[^,]+]], i32* [[OMP_UB:%[^,]+]], i32* [[OMP_ST:%[^,]+]], i32 1, i32 1)
// OMP5-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// OMP5-NEXT: [[UBCMP:%.+]] = icmp sgt i32 [[UB]], 4
// OMP5-NEXT: br i1 [[UBCMP]], label [[UB_TRUE:%[^,]+]], label [[UB_FALSE:%[^,]+]]
// OMP5: [[UBRESULT:%.+]] = phi i32 [ 4, [[UB_TRUE]] ], [ [[UBVAL:%[^,]+]], [[UB_FALSE]] ]
// OMP5-NEXT: store i32 [[UBRESULT]], i32* [[OMP_UB]]
// OMP5-NEXT: [[LB:%.+]] = load i32, i32* [[OMP_LB]]
// OMP5-NEXT: store i32 [[LB]], i32* [[OMP_IV:[^,]+]]
// OMP5-NEXT: br label %[[LOOP1_HEAD:.+]]

// Loop header
// OMP5: [[LOOP1_HEAD]]
// OMP5: [[IV:%.+]] = load i32, i32* [[OMP_IV]]
// OMP5-NEXT: [[UB:%.+]] = load i32, i32* [[OMP_UB]]
// OMP5-NEXT: [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// OMP5-NEXT: br i1 [[CMP]], label %[[LOOP1_BODY:[^,]+]], label %[[LOOP1_END:[^,]+]]
  for (int j = 5 ; j != 0; --j)
// Start of body: calculate i from IV:
// OMP5: [[LOOP1_BODY]]
// OMP5: [[IV2_1:%.+]] = load i32, i32* [[OMP_IV]]
// OMP5-NEXT: [[CALC_II_1:%.+]] = mul nsw i32 [[IV2_1]], 1
// OMP5-NEXT: [[CALC_II_2:%.+]] = sub nsw i32 5, [[CALC_II_1]]
// OMP5-NEXT: store i32 [[CALC_II_2]], i32* [[LC_I:.+]]
// OMP5: [[IV2_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// OMP5-NEXT: [[ADD2_2:%.+]] = add nsw i32 [[IV2_2]], 1
// OMP5-NEXT: store i32 [[ADD2_2]], i32* [[OMP_IV]]
// OMP5-NEXT: br label %[[LOOP1_HEAD]]
    ;
// OMP5: [[LOOP1_END]]
// OMP5: call void @__kmpc_for_static_fini(%struct.ident_t* [[LOOP_LOC]], i32 [[GTID]])
// OMP5-NOT: __kmpc_barrier
  return 0;
// OMP5: ret i32 0
}

// OMP5-LABEL: range_for_single
void range_for_single() {
  int arr[10] = {0};
// OMP5: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* {{.+}}, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [10 x i32]*)* [[OUTLINED:@.+]] to void (i32*, i32*, ...)*), [10 x i32]* %{{.+}})
#pragma omp parallel for
  for (auto &a : arr)
    (void)a;
}

// OMP5: define internal void @.omp_outlined.(i32* {{.+}}, i32* {{.+}}, [10 x i32]* nonnull align 4 dereferenceable(40) %arr)
// OMP5: [[ARR_ADDR:%.+]] = alloca [10 x i32]*,
// OMP5: [[IV:%.+]] = alloca i64,
// OMP5: [[RANGE_ADDR:%.+]] = alloca [10 x i32]*,
// OMP5: [[END_ADDR:%.+]] = alloca i32*,
// OMP5: alloca i32*,
// OMP5: alloca i32*,
// OMP5: alloca i64,
// OMP5: [[BEGIN_INIT:%.+]] = alloca i32*,
// OMP5: [[LB:%.+]] = alloca i64,
// OMP5: [[UB:%.+]] = alloca i64,
// OMP5: [[STRIDE:%.+]] = alloca i64,
// OMP5: [[IS_LAST:%.+]] = alloca i32,
// OMP5: [[BEGIN:%.+]] = alloca i32*,
// OMP5: [[A_PTR:%.+]] = alloca i32*,

// __range = arr;
// OMP5: [[ARR:%.+]] = load [10 x i32]*, [10 x i32]** [[ARR_ADDR]],
// OMP5: store [10 x i32]* [[ARR]], [10 x i32]** [[RANGE_ADDR]],

// __end = end(_range);
// OMP5: [[RANGE:%.+]] = load [10 x i32]*, [10 x i32]** [[RANGE_ADDR]],
// OMP5: [[RANGE_0:%.+]] = getelementptr inbounds [10 x i32], [10 x i32]* [[RANGE]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// OMP5: [[RANGE_10:%.+]] = getelementptr inbounds i32, i32* [[RANGE_0]], i{{[0-9]+}} 10
// OMP5: store i32* [[RANGE_10]], i32** [[END_ADDR]],

// OMP5: [[RANGE:%.+]] = load [10 x i32]*, [10 x i32]** [[RANGE_ADDR]],
// OMP5: [[RANGE_0:%.+]] = getelementptr inbounds [10 x i32], [10 x i32]* [[RANGE]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// OMP5: store i32* [[RANGE_0]], i32** [[CAP1:%.+]],
// OMP5: [[END:%.+]] = load i32*, i32** [[END_ADDR]],
// OMP5: store i32* [[END]], i32** [[CAP2:%.+]],

// calculate number of elements.
// OMP5: [[CAP2_VAL:%.+]] = load i32*, i32** [[CAP2]],
// OMP5: [[CAP1_VAL:%.+]] = load i32*, i32** [[CAP1]],
// OMP5: [[CAP2_I64:%.+]] = ptrtoint i32* [[CAP2_VAL]] to i64
// OMP5: [[CAP1_I64:%.+]] = ptrtoint i32* [[CAP1_VAL]] to i64
// OMP5: [[DIFF:%.+]] = sub i64 [[CAP2_I64]], [[CAP1_I64]]
// OMP5: [[NUM:%.+]] = sdiv exact i64 [[DIFF]], 4
// OMP5: [[NUM1:%.+]] = sub nsw i64 [[NUM]], 1
// OMP5: [[NUM2:%.+]] = add nsw i64 [[NUM1]], 1
// OMP5: [[NUM3:%.+]] = sdiv i64 [[NUM2]], 1
// OMP5: [[NUM4:%.+]] = sub nsw i64 [[NUM3]], 1
// OMP5: store i64 [[NUM4]], i64* [[CAP3:%.+]],
// OMP5: [[RANGE_0:%.+]] = load i32*, i32** [[CAP1]],

// __begin = begin(range);
// OMP5: store i32* [[RANGE_0]], i32** [[BEGIN_INIT]],
// OMP5: [[CAP1_VAL:%.+]] = load i32*, i32** [[CAP1]],
// OMP5: [[CAP2_VAL:%.+]] = load i32*, i32** [[CAP2]],
// OMP5: [[CMP:%.+]] = icmp ult i32* [[CAP1_VAL]], [[CAP2_VAL]]

// __begin >= __end ? goto then : goto exit;
// OMP5: br i1 [[CMP]], label %[[THEN:.+]], label %[[EXIT:.+]]

// OMP5: [[THEN]]:

// lb = 0;
// OMP5: store i64 0, i64* [[LB]],

// ub = number of elements
// OMP5: [[NUM:%.+]] = load i64, i64* [[CAP3]],
// OMP5: store i64 [[NUM]], i64* [[UB]],

// stride = 1;
// OMP5: store i64 1, i64* [[STRIDE]],

// is_last = 0;
// OMP5: store i32 0, i32* [[IS_LAST]],

// loop.
// OMP5: call void @__kmpc_for_static_init_8(%struct.ident_t* {{.+}}, i32 [[GTID:%.+]], i32 34, i32* [[IS_LAST]], i64* [[LB]], i64* [[UB]], i64* [[STRIDE]], i64 1, i64 1)

// ub = (ub > number_of_elems ? number_of_elems : ub);
// OMP5: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// OMP5: [[NUM_VAL:%.+]] = load i64, i64* [[CAP3]],
// OMP5: [[CMP:%.+]] = icmp sgt i64 [[UB_VAL]], [[NUM_VAL]]
// OMP5: br i1 [[CMP]], label %[[TRUE:.+]], label %[[FALSE:.+]]

// OMP5: [[TRUE]]:
// OMP5: [[NUM_VAL:%.+]] = load i64, i64* [[CAP3]],
// OMP5: br label %[[END:.+]]

// OMP5: [[FALSE]]:
// OMP5: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// OMP5: br label %[[END:.+]]

// OMP5: [[END]]:
// OMP5: [[MIN:%.+]] = phi i64 [ [[NUM_VAL]], %[[TRUE]] ], [ [[UB_VAL]], %[[FALSE]] ]
// OMP%: store i64 [[MIN]], i64* [[UB]],

// iv = lb;
// OMP5: [[LB_VAL:%.+]] = load i64, i64* [[LB]],
// OMP5: store i64 [[LB_VAL]], i64* [[IV]],

// goto loop;
// loop:
// OMP5: br label %[[LOOP:.+]]

// OMP5: [[LOOP]]:

// iv <= ub ? goto body : goto end;
// OMP5: [[IV_VAL:%.+]] = load i64, i64* [[IV]],
// OMP5: [[UB_VAL:%.+]] = load i64, i64* [[UB]],
// OMP5: [[CMP:%.+]] = icmp sle i64 [[IV_VAL]], [[UB_VAL]]
// OMP5: br i1 [[CMP]], label %[[BODY:.+]], label %[[END:.+]]

// body:
// __begin = begin(arr) + iv * 1;
// OMP5: [[BODY]]:
// OMP5: [[CAP1_VAL:%.+]] = load i32*, i32** [[CAP1]],
// OMP5: [[IV_VAL:%.+]] = load i64, i64* [[IV]],
// OMP5: [[MUL:%.+]] = mul nsw i64 [[IV_VAL]], 1
// OMP5: [[ADDR:%.+]] = getelementptr inbounds i32, i32* [[CAP1_VAL]], i64 [[MUL]]
// OMP5: store i32* [[ADDR]], i32** [[BEGIN]],

// a = *__begin;
// OMP5: [[BEGIN_VAL:%.+]] = load i32*, i32** [[BEGIN]],
// OMP5: store i32* [[BEGIN_VAL]], i32** [[A_PTR]],

// (void)a;
// OMP5: load i32*, i32** [[A_PTR]],

// iv += 1;
// OMP5: [[IV_VAL:%.+]] = load i64, i64* [[IV]],
// OMP5: [[IV_VAL_ADD_1:%.+]] = add nsw i64 [[IV_VAL]], 1
// OMP5: store i64 [[IV_VAL_ADD_1]], i64* [[IV]],

// goto loop;
// OMP5: br label %[[LOOP]]

// end:
// OMP5: [[END]]:
// OMP5: call void @__kmpc_for_static_fini(%struct.ident_t* {{.+}}, i32 [[GTID:%.+]])
// exit:
// OMP5: [[EXIT]]:
// OMP5: ret void

// OMP5-LABEL: range_for_collapsed
void range_for_collapsed() {
  int arr[10] = {0};
// OMP5: call void @__kmpc_for_static_init_8(%struct.ident_t* {{.+}}, i32 [[GTID:%.+]], i32 34, i32* %{{.+}}, i64* %{{.+}}, i64* %{{.+}}, i64* %{{.+}}, i64 1, i64 1)
#pragma omp parallel for collapse(2)
  for (auto &a : arr)
    for (auto b : arr)
      a = b;
// OMP5: call void @__kmpc_for_static_fini(%struct.ident_t* {{.+}}, i32 [[GTID:%.+]])
}
#endif // OMP5

#endif // HEADER

