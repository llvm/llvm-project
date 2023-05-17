// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple %itanium_abi_triple -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple %itanium_abi_triple -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef __INTPTR_TYPE__ intptr_t;

// CHECK-DAG: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, ptr }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr constant [[IDENT_T_TY]] { i32 0, i32 2, i32 0, i32 22, ptr [[STR]] }

void foo();

struct S {
  intptr_t a, b, c;
  S(intptr_t a) : a(a) {}
  operator char() { return a; }
  ~S() {}
};

template <typename T>
T tmain() {
#pragma omp parallel proc_bind(master)
  foo();
  return T();
}

int main() {
#pragma omp parallel proc_bind(spread)
  foo();
#pragma omp parallel proc_bind(close)
  foo();
  return tmain<int>();
}

// CHECK-LABEL: @main
// CHECK:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEF_LOC_2]])
// CHECK:       call {{.*}}void @__kmpc_push_proc_bind(ptr [[DEF_LOC_2]], i32 [[GTID]], i32 4)
// CHECK:       call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// CHECK:       call {{.*}}void @__kmpc_push_proc_bind(ptr [[DEF_LOC_2]], i32 [[GTID]], i32 3)
// CHECK:       call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call(

// CHECK-LABEL: @{{.+}}tmain
// CHECK:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEF_LOC_2]])
// CHECK:       call {{.*}}void @__kmpc_push_proc_bind(ptr [[DEF_LOC_2]], i32 [[GTID]], i32 2)
// CHECK:       call {{.*}}void (ptr, i32, ptr, ...) @__kmpc_fork_call(
// CHECK:       ret i32 0
// CHECK-NEXT:  }

#endif
