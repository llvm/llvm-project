// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr global %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-LABEL: define {{.*void}} @{{.*}}static_not_chunked{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void static_not_chunked(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(static) ordered
// CHECK: call void @__kmpc_dispatch_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 66, i32 0, i32 4571423, i32 1, i32 1)
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
  for (int i = 32000000; i > 33; i += -7) {
// CHECK: [[LOOP1_BODY]]
// Start of body: calculate i from IV:
// CHECK: [[IV1_1:%.+]] = load i32, i32* [[OMP_IV]]
// CHECK-NEXT: [[CALC_I_1:%.+]] = mul nsw i32 [[IV1_1]], 7
// CHECK-NEXT: [[CALC_I_2:%.+]] = sub nsw i32 32000000, [[CALC_I_1]]
// CHECK-NEXT: store i32 [[CALC_I_2]], i32* [[LC_I:.+]]

// ... start of ordered region ...
// CHECK-NEXT: call void @__kmpc_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.access.group
// CHECK-NEXT: call void @__kmpc_end_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... end of ordered region ...
    #pragma omp ordered
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]
// CHECK-NEXT: call void @__kmpc_dispatch_fini_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

// CHECK-LABEL: define {{.*void}} @{{.*}}dynamic1{{.*}}(float* {{.+}}, float* {{.+}}, float* {{.+}}, float* {{.+}})
void dynamic1(float *a, float *b, float *c, float *d) {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:[@%].+]])
  #pragma omp for schedule(dynamic) ordered
// CHECK: call void @__kmpc_dispatch_init_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 67, i64 0, i64 16908287, i64 1, i64 1)
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

// ... start of ordered region ...
// CHECK-NEXT: call void @__kmpc_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.access.group
// CHECK-NEXT: call void @__kmpc_end_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... end of ordered region ...
    #pragma omp ordered threads
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]

// ... end iteration for ordered loop ...
// CHECK-NEXT: call void @__kmpc_dispatch_fini_8u([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
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
  #pragma omp for schedule(auto) collapse(2) ordered
// CHECK: call void @__kmpc_dispatch_init_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 70, i64 0, i64 [[LAST_ITER:%[^,]+]], i64 1, i64 1)
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

// ... start of ordered region ...
// CHECK: call void @__kmpc_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.access.group
// CHECK-NEXT: call void @__kmpc_end_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... end of ordered region ...
    #pragma omp ordered
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i64, i64* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i64 [[IV1_2]], 1
// CHECK-NEXT: store i64 [[ADD1_2]], i64* [[OMP_IV]]

// ... end iteration for ordered loop ...
// CHECK-NEXT: call void @__kmpc_dispatch_fini_8([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
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
  #pragma omp for collapse(2) schedule(runtime) ordered
// CHECK: call void @__kmpc_dispatch_init_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 69, i32 0, i32 199, i32 1, i32 1)
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

// ... start of ordered region ...
// CHECK: call void @__kmpc_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... loop body ...
// End of body: store into a[i]:
// CHECK: store float [[RESULT:%.+]], float* {{%.+}}
// CHECK-NOT: !llvm.access.group
// CHECK-NEXT: call void @__kmpc_end_ordered([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// ... end of ordered region ...
    #pragma omp ordered threads
    a[i] = b[i] * c[i] * d[i];
// CHECK: [[IV1_2:%.+]] = load i32, i32* [[OMP_IV]]{{.*}}
// CHECK-NEXT: [[ADD1_2:%.+]] = add nsw i32 [[IV1_2]], 1
// CHECK-NEXT: store i32 [[ADD1_2]], i32* [[OMP_IV]]

// ... end iteration for ordered loop ...
// CHECK-NEXT: call void @__kmpc_dispatch_fini_4([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT: br label %{{.+}}
  }
// CHECK: [[LOOP1_END]]
// CHECK: [[O_LOOP1_END]]
// CHECK: call {{.+}} @__kmpc_barrier([[IDENT_T_TY]]* [[IMPLICIT_BARRIER_LOC]], i32 [[GTID]])
// CHECK: ret void
}

float f[10];
// CHECK-LABEL: foo_simd
void foo_simd(int low, int up) {
  // CHECK: store float 0.000000e+00, float* %{{.+}}, align {{[0-9]+}}, !llvm.access.group !
  // CHECK-NEXT: call void [[CAP_FUNC:@.+]](i32* %{{.+}}), !llvm.access.group !
#pragma omp simd
  for (int i = low; i < up; ++i) {
    f[i] = 0.0;
#pragma omp ordered simd
    f[i] = 1.0;
  }
  // CHECK: store float 0.000000e+00, float* %{{.+}}, align {{[0-9]+}}
  // CHECK-NEXT: call void [[CAP_FUNC:@.+]](i32* %{{.+}})
#pragma omp for simd ordered
  for (int i = low; i < up; ++i) {
    f[i] = 0.0;
#pragma omp ordered simd
    f[i] = 1.0;
  }
}

// CHECK: define internal void [[CAP_FUNC]](i32* dereferenceable({{[0-9]+}}) %{{.+}}) #
// CHECK: store float 1.000000e+00, float* %{{.+}}, align
// CHECK-NEXT: ret void

#endif // HEADER

