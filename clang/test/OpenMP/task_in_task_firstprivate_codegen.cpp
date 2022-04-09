// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: @main
int main() {
  double var = 0;
  // Check that var is firstprivatized in the outermost task.
  // CHECK: [[BASE:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 {{.+}}, i32 1, i64 48, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[TASK_TY:%.+]]*)* @{{.+}} to i32 (i32, i8*)*))
  // CHECK: [[TD:%.+]] = bitcast i8* [[BASE]] to [[TASK_TY]]*
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds [[TASK_TY]], [[TASK_TY]]* [[TD]], i32 0, i32 1
  // CHECK: [[VAR_FP:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[PRIVS]], i32 0, i32 0
  // CHECK: [[VAR_VAL:%.+]] = load double, double* %{{.+}},
  // CHECK: store double [[VAR_VAL]], double* [[VAR_FP]],
  // CHECK: call i32 @__kmpc_omp_task(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[BASE]])
#pragma omp task
#pragma omp task
  var += 1;
  return 0;
}

#endif
