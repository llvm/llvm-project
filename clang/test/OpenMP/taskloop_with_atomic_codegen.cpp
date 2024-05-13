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

// CHECK-LABEL: @main
int main() {
  unsigned occupanices = 0;

// CHECK: call void @__kmpc_taskloop(ptr @{{.+}}, i32 %{{.+}}, ptr %{{.+}}, i32 1, ptr %{{.+}}, ptr %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, ptr null)
#pragma omp taskloop
  for (int i = 0; i < 1; i++) {
#pragma omp atomic
    occupanices++;
  }
}

// CHECK: define internal noundef i32 @{{.+}}(
// Check that occupanices var is firstprivatized.
// CHECK-DAG: atomicrmw add ptr [[FP_OCCUP:%.+]], i32 1 monotonic, align 4
// CHECK-DAG: [[FP_OCCUP]] = load ptr, ptr [[FP_OCCUP_ADDR:%[^,]+]],
// CHECK-DAG: call void %{{.*}}(ptr %{{.+}}, ptr [[FP_OCCUP_ADDR]])

#endif
