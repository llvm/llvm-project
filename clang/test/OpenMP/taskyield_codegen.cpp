// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp taskyield
  return a + argc;
}

// CHECK-LABEL: @main
int main(int argc, char **argv) {
  static int a;
#pragma omp taskyield
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* @{{.+}})
  // CHECK: call i32 @__kmpc_omp_taskyield([[IDENT_T]]* @{{.+}}, i32 [[GTID]], i32 0)
  // CHECK: call {{.+}} [[TMAIN_INT:@.+]](i{{[0-9][0-9]}}
  // CHECK: call {{.+}} [[TMAIN_CHAR:@.+]](i{{[0-9]}}
  return tmain(argc) + tmain(argv[0][0]) + a;
}

// CHECK: define {{.+}} [[TMAIN_INT]](
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* @{{.+}})
// CHECK: call i32 @__kmpc_omp_taskyield([[IDENT_T]]* @{{.+}}, i32 [[GTID]], i32 0)

// CHECK: define {{.+}} [[TMAIN_CHAR]](
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num([[IDENT_T]]* @{{.+}})
// CHECK: call i32 @__kmpc_omp_taskyield([[IDENT_T]]* @{{.+}}, i32 [[GTID]], i32 0)

#endif
