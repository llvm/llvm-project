// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[PRIVATES:%.+]] = type { i8*, i8* }

struct S {
  int a;
  S() : a(0) {}
  S(const S&) {}
  S& operator=(const S&) {return *this;}
  ~S() {}
  friend S operator+(const S&a, const S&b) {return a;}
};


int main(int argc, char **argv) {
  int a;
  float b;
  S c[5];
  short d[argc];
#pragma omp taskgroup task_reduction(+: a, b, argc)
  {
#pragma omp taskgroup task_reduction(-:c, d)
#pragma omp parallel
#pragma omp taskloop in_reduction(+:a) in_reduction(-:d)
    for (int i = 0; i < 5; ++i)
      a += d[a];
  }
  return 0;
}

// CHECK-LABEL: @main
// CHECK:       void @__kmpc_taskgroup(%struct.ident_t* @1, i32 [[GTID:%.+]])
// CHECK:       [[TD1:%.+]] = call i8* @__kmpc_taskred_init(i32 [[GTID]], i32 3, i8* %
// CHECK-NEXT:  store i8* [[TD1]], i8** [[TD1_ADDR:%[^,]+]],
// CHECK-NEXT:  call void @__kmpc_taskgroup(%struct.ident_t* @1, i32 [[GTID]])
// CHECK:       [[TD2:%.+]] = call i8* @__kmpc_taskred_init(i32 [[GTID]], i32 2, i8* %
// CHECK-NEXT:  store i8* [[TD2]], i8** [[TD2_ADDR:%[^,]+]],
// CHECK-NEXT:  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @1, i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i64, i16*, i8**, i8**)* [[OMP_PARALLEL:@.+]] to void (i32*, i32*, ...)*), i32* %{{.+}}, i64 %{{.+}}, i16* %{{.+}}, i8** [[TD1_ADDR]], i8** [[TD2_ADDR]])
// CHECK-NEXT:  call void @__kmpc_end_taskgroup(%struct.ident_t* @1, i32 [[GTID]])
// CHECK-NEXT:  call void @__kmpc_end_taskgroup(%struct.ident_t* @1, i32 [[GTID]])

// CHECK:       define internal void [[OMP_PARALLEL]](
// CHECK:       [[TASK_T:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @1, i32 [[GTID:%.+]], i32 1, i64 96, i64 40, i32 (i32, i8*)* bitcast (i32 (i32, [[T:%.+]]*)* [[OMP_TASK:@.+]] to i32 (i32, i8*)*))
// CHECK-NEXT:  [[TASK_T_WITH_PRIVS:%.+]] = bitcast i8* [[TASK_T]] to [[T]]*
// CHECK:       [[PRIVS:%.+]] = getelementptr inbounds [[T]], [[T]]* [[TASK_T_WITH_PRIVS]], i32 0, i32 1
// CHECK:       [[TD1_REF:%.+]] = getelementptr inbounds [[PRIVATES]], [[PRIVATES]]* [[PRIVS]], i32 0, i32 0
// CHECK-NEXT:  [[TD1:%.+]] = load i8*, i8** %{{.+}},
// CHECK-NEXT:  store i8* [[TD1]], i8** [[TD1_REF]],
// CHECK-NEXT:  [[TD2_REF:%.+]] = getelementptr inbounds [[PRIVATES]], [[PRIVATES]]* [[PRIVS]], i32 0, i32 1
// CHECK-NEXT:  [[TD2:%.+]] = load i8*, i8** %{{.+}},
// CHECK-NEXT:  store i8* [[TD2]], i8** [[TD2_REF]],
// CHECK:       call void @__kmpc_taskloop(%struct.ident_t* @1, i32 [[GTID]], i8* [[TASK_T]], i32 1,
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal {{.*}} [[OMP_TASK]](
// CHECK:       [[FN:%.+]] = bitcast void (i8*, ...)* %{{.*}} to void (i8*,
// CHECK:       call void [[FN]](i8* %{{.+}}, i8*** [[TD1_REF:%[^,]+]], i8*** [[TD2_REF:%[^,]+]])
// CHECK-NEXT:  [[TD1_ADDR:%.+]] = load i8**, i8*** [[TD1_REF]],
// CHECK-NEXT:  [[TD2_ADDR:%.+]] = load i8**, i8*** [[TD2_REF]],
// CHECK-NEXT:  [[A_REF:%.+]] = getelementptr inbounds %
// CHECK-NEXT:  [[A_ADDR:%.+]] = load i32*, i32** [[A_REF]],
// CHECK-NEXT:  [[TD1:%.+]] = load i8*, i8** [[TD1_ADDR]],
// CHECK-NEXT:  [[GTID:%.+]] = load i32, i32* %
// CHECK-NEXT:  [[A_PTR:%.+]] = bitcast i32* [[A_ADDR]] to i8*
// CHECK-NEXT:  call i8* @__kmpc_task_reduction_get_th_data(i32 [[GTID]], i8* [[TD1]], i8* [[A_PTR]])
// CHECK:       [[D_REF:%.+]] = getelementptr inbounds %
// CHECK-NEXT:  [[D_ADDR:%.+]] = load i16*, i16** [[D_REF]],
// CHECK:       [[TD2:%.+]] = load i8*, i8** [[TD2_ADDR]],
// CHECK-NEXT:  [[D_PTR:%.+]] = bitcast i16* [[D_ADDR]] to i8*
// CHECK-NEXT:  call i8* @__kmpc_task_reduction_get_th_data(i32 [[GTID]], i8* [[TD2]], i8* [[D_PTR]])
// CHECK:       add nsw i32
// CHECK:       store i32 %
#endif
