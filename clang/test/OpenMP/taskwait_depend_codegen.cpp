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

void foo() {}

template <class T>
T tmain(T &argc) {
  static T a;
  #pragma omp taskwait depend(in:argc)
  return a + argc;
}
int main(int argc, char **argv) {
  int n = 0;
  #pragma omp task shared(n,argc) depend(out:n)
     n = argc;
  return tmain(n);
}

// CHECK-LABEL: @main
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%{{.+}}* @{{.+}})
// CHECK: [[ALLOC:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i64 40, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, %{{.+}}*)* @{{.+}} to i32 (i32, i8*)*))
// CHECK: %{{.+}} = call i32 @__kmpc_omp_task_with_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ALLOC]], i32 1, i8* %{{[0-9]*}}, i32 0, i8* null)

// CHECK-LABEL: tmain
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%{{.+}}* @{{.+}})
// CHECK: call void @__kmpc_omp_wait_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i8* %{{.}}, i32 0, i8* null)


#endif
