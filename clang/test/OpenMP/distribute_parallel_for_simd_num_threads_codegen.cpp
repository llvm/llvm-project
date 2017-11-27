// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple %itanium_abi_triple -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple %itanium_abi_triple -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
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
  operator char() { return a; }
  ~S() {}
};

template <typename T, int C>
int tmain() {
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd num_threads(C)
  for (int i = 0; i < 100; i++)
    foo();
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for simd num_threads(T(23))
  for (int i = 0; i < 100; i++)
    foo();
  return 0;
}

int main() {
  S s(0);
  char a = s;
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_0:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[OFFLOADING_FUN_1:@.+]](
// CHECK: invoke{{.+}} [[TMAIN_5:@.+]]()
// CHECK: invoke{{.+}} [[TMAIN_1:@.+]]()
#pragma omp target
#pragma omp teams
  // CHECK: define internal void [[OFFLOADING_FUN_0]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_TEAMS_OUTLINED_0:@.+]] to {{.+}})
#pragma omp distribute parallel for simd num_threads(2)
  for (int i = 0; i < 100; i++) {
    // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_0]](
    // CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 2)
    // CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
    foo();
  }
#pragma omp target
#pragma omp teams
  // CHECK: define internal void [[OFFLOADING_FUN_1]](

  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}}* [[OMP_TEAMS_OUTLINED_1:@.+]] to {{.+}})
#pragma omp distribute parallel for simd num_threads(a)
  for (int i = 0; i < 100; i++) {
    // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_1]](
    // CHECK-DAG: [[A_ADDR:%.+]] = alloca i8*,
    // CHECK-DAG: [[A_REF:%.+]] = load i8*, i8** [[A_ADDR]],
    // CHECK-DAG: [[A_VAL:%.+]] = load i8, i8* [[A_REF]],
    // CHECK-DAG: [[A_EXT:%.+]] = sext i8 [[A_VAL]] to {{.+}}
    // CHECK: call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 [[A_EXT]])
    // CHECK: call {{.*}}void {{.*}} @__kmpc_fork_call(
    foo();
  }
  return a + tmain<char, 5>() + tmain<S, 1>();
}

// tmain 5
// CHECK-DAG: define {{.*}}i{{[0-9]+}} [[TMAIN_5]]()
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[T_OFFLOADING_FUN_0:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[T_OFFLOADING_FUN_1:@.+]](

// tmain 1
// CHECK-DAG: define {{.*}}i{{[0-9]+}} [[TMAIN_1]]()
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[T_OFFLOADING_FUN_2:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target_teams(
// CHECK: call void [[T_OFFLOADING_FUN_3:@.+]](

// CHECK: define internal void [[T_OFFLOADING_FUN_0]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[T_OMP_TEAMS_OUTLINED_0:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[T_OMP_TEAMS_OUTLINED_0]](
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 5)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(

// CHECK: define internal void [[T_OFFLOADING_FUN_1]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[T_OMP_TEAMS_OUTLINED_1:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[T_OMP_TEAMS_OUTLINED_1]](
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 23)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(

// CHECK: define internal void [[T_OFFLOADING_FUN_2]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[T_OMP_TEAMS_OUTLINED_2:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[T_OMP_TEAMS_OUTLINED_2]](
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 1)
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(

// CHECK: define internal void [[T_OFFLOADING_FUN_3]](
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[T_OMP_TEAMS_OUTLINED_3:@.+]] to {{.+}})

// CHECK: define{{.+}} void [[T_OMP_TEAMS_OUTLINED_3]](
// CHECK-DAG:   [[CALL_RES:%.+]] = invoke{{.*}} i8 [[S_TY_CHAR_OP:@.+]]([[S_TY]]* {{.+}})
// CHECK-DAG:   [[CALL_RES_SEXT:%.+]] = sext i8 [[CALL_RES]] to {{.+}}
// CHECK:       call {{.*}}void @__kmpc_push_num_threads([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 {{.+}}, i32 [[CALL_RES_SEXT]])
// CHECK:       call {{.*}}void {{.*}} @__kmpc_fork_call(
#endif
