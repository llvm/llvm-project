// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void fn1();
void fn2();
void fn3();
void fn4();
void fn5();
void fn6();

int Arg;

// CHECK-LABEL: define {{.*}}void @{{.+}}gtid_test
void gtid_test() {
#pragma omp target
// CHECK: call i{{[0-9]+}} @__tgt_target(
// CHECK: call void [[OFFLOADING_FUN_0:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target(
// CHECK: call void [[OFFLOADING_FUN_1:@.+]](
#pragma omp teams distribute parallel for
  for(int i = 0 ; i < 100; i++) {}
  // CHECK: define internal void [[OFFLOADING_FUN_0]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_TEAMS_OUTLINED_0:@.+]] to {{.+}})
  // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_0]](
  // CHECK: call void @__kmpc_for_static_init_4(
  // CHECK:  call void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 2, {{.+}}* [[OMP_OUTLINED_0:@.+]] to void
  // CHECK: call void @__kmpc_for_static_fini(

  // CHECK: define{{.+}} void [[OMP_OUTLINED_0]](
  // CHECK: call void @__kmpc_for_static_init_4(
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK: ret
#pragma omp target
#pragma omp teams distribute parallel for if (parallel: false)
  for(int i = 0 ; i < 100; i++) {
  // CHECK: define internal void [[OFFLOADING_FUN_1]](
  // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_TEAMS_OUTLINED_1:@.+]] to {{.+}})
  // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_1]](
  // CHECK: call void @__kmpc_for_static_init_4(
  // CHECK: call void @__kmpc_serialized_parallel(
  // CHECK: call void [[OMP_OUTLINED_1:@.+]](
  // CHECK: call void @__kmpc_end_serialized_parallel(
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK: define{{.+}} void [[OMP_OUTLINED_1]](
  // CHECK: call void @__kmpc_for_static_init_4(
  // CHECK: call void @{{.+}}gtid_test
  // CHECK: call void @__kmpc_for_static_fini(
  // CHECK: ret
    gtid_test();
  }
}


template <typename T>
int tmain(T Arg) {
#pragma omp target
#pragma omp teams distribute parallel for if (true)
  for(int i = 0 ; i < 100; i++) {
    fn1();
  }
#pragma omp target
#pragma omp teams distribute parallel for if (false)
  for(int i = 0 ; i < 100; i++) {
    fn2();
  }
#pragma omp target
#pragma omp teams distribute parallel for if (parallel: Arg)
  for(int i = 0 ; i < 100; i++) {
    fn3();
  }
  return 0;
}

// CHECK-LABEL: define {{.*}}i{{[0-9]+}} @main()
int main() {
// CHECK: call i{{[0-9]+}} @__tgt_target(
// CHECK: call void [[OFFLOADING_FUN_0:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target(
// CHECK: call void [[OFFLOADING_FUN_1:@.+]](
// CHECK: call i{{[0-9]+}} @__tgt_target(
// CHECK: call void [[OFFLOADING_FUN_2:@.+]](
// CHECK: = call {{.*}}i{{.+}} @{{.+}}tmain
#pragma omp target
#pragma omp teams distribute parallel for if (true)
  for(int i = 0 ; i < 100; i++) {
    // CHECK: define internal void [[OFFLOADING_FUN_0]](
    // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_TEAMS_OUTLINED_0:@.+]] to {{.+}})
    // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_0]](

    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 2, {{.+}}* [[OMP_OUTLINED_2:@.+]] to void
    // CHECK: call void @__kmpc_for_static_fini(
    // CHECK: define{{.+}} void [[OMP_OUTLINED_2]](
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call {{.*}}void @{{.+}}fn4
    // CHECK: call void @__kmpc_for_static_fini(

    fn4();
  }

#pragma omp target
#pragma omp teams distribute parallel for if (false)
  for(int i = 0 ; i < 100; i++) {
    // CHECK: define internal void [[OFFLOADING_FUN_1]](
    // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 0, {{.+}}* [[OMP_TEAMS_OUTLINED_1:@.+]] to {{.+}})
    // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_1]](

    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call void @__kmpc_serialized_parallel(
    // CHECK: call void [[OMP_OUTLINED_3:@.+]](
    // CHECK: call void @__kmpc_end_serialized_parallel(
    // CHECK: call void @__kmpc_for_static_fini(

    // CHECK: define{{.+}} void [[OMP_OUTLINED_3]](
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call {{.*}}void @{{.+}}fn5
    // CHECK: call void @__kmpc_for_static_fini(
    fn5();
  }

#pragma omp target
#pragma omp teams distribute parallel for if (Arg)
  for(int i = 0 ; i < 100; i++) {
    // CHECK: define internal void [[OFFLOADING_FUN_2]](
    // CHECK: call {{.*}}void {{.+}} @__kmpc_fork_teams({{.+}}, i32 1, {{.+}}* [[OMP_TEAMS_OUTLINED_2:@.+]] to {{.+}})
    // CHECK: define{{.+}} void [[OMP_TEAMS_OUTLINED_2]](

    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 2, {{.+}}* [[OMP_OUTLINED_4:@.+]] to void
    // CHECK: call void @__kmpc_serialized_parallel(
    // CHECK: call void [[OMP_OUTLINED_4:@.+]](
    // CHECK: call void @__kmpc_end_serialized_parallel(
    // CHECK: call void @__kmpc_for_static_fini(

    // CHECK: define{{.+}} void [[OMP_OUTLINED_4]](
    // CHECK: call void @__kmpc_for_static_init_4(
    // CHECK: call {{.*}}void @{{.+}}fn6
    // CHECK: call void @__kmpc_for_static_fini(
    fn6();
  }

  return tmain(Arg);
}

// CHECK-LABEL: define {{.+}} @{{.+}}tmain

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 2, void {{.+}}* [[T_OUTLINE_FUN_1:@.+]] to void
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: define internal {{.*}}void [[T_OUTLINE_FUN_1]]
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void @{{.+}}fn1
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(
// CHECK: call void [[T_OUTLINE_FUN_2:@.+]](
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(
// CHECK: call void @__kmpc_for_static_fini(

// CHECK: define internal {{.*}}void [[T_OUTLINE_FUN_2]]
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void @{{.+}}fn2
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void

// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 2, void {{.+}}* [[T_OUTLINE_FUN_3:@.+]] to void
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(
// call void [[T_OUTLINE_FUN_3:@.+]](
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(

// CHECK: define internal {{.*}}void [[T_OUTLINE_FUN_3]]
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}void @{{.+}}fn3
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: ret void
#endif
