// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=51 -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=51 -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void foo() {}

template <class T>
T nmain(T &argc) {
  static T a;
  #pragma omp taskwait depend(in:argc) nowait
  return a + argc;
}
int main(int argc, char **argv) {
  int n = 0;
  n = argc;
  return nmain(n);
}

// CHECK-LABEL: @main{{.*}}

// CHECK-LABEL: nmain
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr @{{.+}})
// CHECK: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 [[GTID]], i32 1, ptr %{{.}}, i32 0, ptr null, i32 1)


#endif
