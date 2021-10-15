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

// CHECK-DAG: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[S_TY:%.+]] = type { [[INTPTR_T_TY:i[0-9]+]], [[INTPTR_T_TY]], [[INTPTR_T_TY]] }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr constant [[IDENT_T_TY]] { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void foo();

struct S {
  intptr_t a, b, c;
  S(intptr_t a) : a(a) {}
  operator char() { extern void mayThrow(); mayThrow(); return a; }
  ~S() {}
};

template <typename T, int C>
int tmain() {
#pragma omp parallel num_threads(C)
  foo();
#pragma omp parallel num_threads(T(23))
  foo();
  return 0;
}

int main() {
  S s(0);
  char a = s;
#pragma omp parallel num_threads(2)
  foo();
#pragma omp parallel num_threads(a)
  foo();
  return a + tmain<char, 5>() + tmain<S, 1>();
}

// CHECK-LABEL: define {{.*}}i{{[0-9]+}} @main()
// CHECK-DAG:   [[S_ADDR:%.+]] = alloca [[S_TY]]
// CHECK-DAG:   [[A_ADDR:%.+]] = alloca i8
// CHECK-DAG:   [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEF_LOC_2]])
// CHECK-DAG:   call {{.*}} [[S_TY_CONSTR:@.+]]([[S_TY]]* {{[^,]*}} [[S_ADDR]], [[INTPTR_T_TY]] noundef [[INTPTR_T_TY_ATTR:(signext )?]]0)
// CHECK:       [[S_CHAR_OP:%.+]] = invoke{{.*}} i8 [[S_TY_CHAR_OP:@.+]]([[S_TY]]* {{[^,]*}} [[S_ADDR]])
// CHECK:       store i8 [[S_CHAR_OP]], i8* [[A_ADDR]]
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 2)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       [[A_VAL:%.+]] = load i8, i8* [[A_ADDR]]
// CHECK:       [[RES:%.+]] = sext i8 [[A_VAL]] to i32
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 [[RES]])
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       invoke{{.*}} [[INT_TY:i[0-9]+]] [[TMAIN_CHAR_5:@.+]]()
// CHECK:       invoke{{.*}} [[INT_TY]] [[TMAIN_S_1:@.+]]()
// CHECK:       call {{.*}} [[S_TY_DESTR:@.+]]([[S_TY]]* {{[^,]*}} [[S_ADDR]])
// CHECK:       ret [[INT_TY]]
// CHECK:       }

// CHECK:       define{{.*}} [[INT_TY]] [[TMAIN_CHAR_5]]()
// CHECK:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEF_LOC_2]])
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 5)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 23)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       ret [[INT_TY]] 0
// CHECK-NEXT:  }

// CHECK:       define{{.*}} [[INT_TY]] [[TMAIN_S_1]]()
// CHECK:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEF_LOC_2]])
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 1)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       {{(invoke|call)}} {{.*}} [[S_TY_CONSTR]]([[S_TY]]* {{[^,]*}} [[S_TEMP:%.+]], [[INTPTR_T_TY]] noundef [[INTPTR_T_TY_ATTR]]23)
// CHECK:       [[S_CHAR_OP:%.+]] = invoke{{.*}} i8 [[S_TY_CHAR_OP]]([[S_TY]]* {{[^,]*}} [[S_TEMP]])
// CHECK:       [[RES:%.+]] = sext {{.*}}i8 [[S_CHAR_OP]] to i32
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 [[RES]])
// CHECK:       {{(invoke|call)}} {{.*}} [[S_TY_DESTR]]([[S_TY]]* {{[^,]*}} [[S_TEMP]])
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
// CHECK:       ret [[INT_TY]] 0
// CHECK:       }

#endif
