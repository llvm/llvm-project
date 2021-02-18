// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=50 -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=50 -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-LABEL: @main
int main() {
  unsigned occupanices = 0;

// CHECK: call void @__kmpc_taskloop(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* %{{.+}}, i32 1, i64* %{{.+}}, i64* %{{.+}}, i64 %{{.+}}, i32 1, i32 0, i64 0, i8* null)
#pragma omp taskloop
  for (int i = 0; i < 1; i++) {
#pragma omp atomic
    occupanices++;
  }
}

// CHECK: define internal i32 @{{.+}}(
// Check that occupanices var is firstprivatized.
// CHECK-DAG: atomicrmw add i32* [[FP_OCCUP:%.+]], i32 1 monotonic, align 4
// CHECK-DAG: [[FP_OCCUP]] = load i32*, i32** [[FP_OCCUP_ADDR:%[^,]+]],
// CHECK-DAG: call void (i8*, ...) %{{.+}}(i8* %{{.+}}, i32** [[FP_OCCUP_ADDR]])

#endif
