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

// CHECK: @main
int main() {
  double var = 0;
  // Check that var is firstprivatized in the outermost task.
  // CHECK: [[BASE:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 {{.+}}, i32 1, i64 48, i64 1, ptr @{{.+}})
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds %{{.+}}, ptr [[BASE]], i32 0, i32 1
  // CHECK: [[VAR_FP:%.+]] = getelementptr inbounds %{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CHECK: [[VAR_VAL:%.+]] = load double, ptr %{{.+}},
  // CHECK: store double [[VAR_VAL]], ptr [[VAR_FP]],
  // CHECK: call i32 @__kmpc_omp_task(ptr @{{.+}}, i32 %{{.+}}, ptr [[BASE]])
#pragma omp task
#pragma omp task
  var += 1;
  return 0;
}

#endif
