// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=51 -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
//
// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=51 -fopenmp-enable-irbuilder -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=51 -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp taskwait
  return a + argc;
}

template <class T>
T no_wait(T argc) {
  static T a;
#pragma omp taskwait nowait
  return a + argc;
}

int main(int argc, char **argv) {
#pragma omp taskwait
  return tmain(argc)+no_wait(argc);
}

// CHECK-LABEL: @main
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%{{.+}}* @{{.+}})
// CHECK: call i32 @__kmpc_omp_taskwait_51(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 0)

// CHECK-LABEL: define {{.*}} @{{.*}}tmain{{.*}}
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%{{.+}}* @{{.+}})
// CHECK: call i32 @__kmpc_omp_taskwait_51(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 0)
//
// CHECK-LABEL: define {{.*}} @{{.*}}no_wait{{.*}}
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%{{.+}}* @{{.+}})
// CHECK: call i32 @__kmpc_omp_taskwait_51(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1)


#endif
