// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple %itanium_abi_triple -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple %itanium_abi_triple -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
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
// CHECK:  call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, {{.+}}* [[GTID_TEST_REGION1:@.+]] to void
#pragma omp parallel
#pragma omp parallel if (parallel: false)
  gtid_test();
// CHECK: ret void
}

// CHECK: define internal {{.*}}void [[GTID_TEST_REGION1]](i{{.+}}* noalias [[GTID_PARAM:%.+]], i32* noalias
// CHECK: store i{{[0-9]+}}* [[GTID_PARAM]], i{{[0-9]+}}** [[GTID_ADDR_REF:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[GTID_ADDR]]
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]])
// CHECK: [[GTID_ADDR:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[GTID_ADDR_REF]]
// CHECK: call void [[GTID_TEST_REGION2:@.+]](i{{[0-9]+}}* [[GTID_ADDR]]
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]])
// CHECK: ret void

// CHECK: define internal {{.*}}void [[GTID_TEST_REGION2]](
// CHECK: call {{.*}}void @{{.+}}gtid_test
// CHECK: ret void

template <typename T>
int tmain(T Arg) {
#pragma omp parallel if (true)
  fn1();
#pragma omp parallel if (false)
  fn2();
#pragma omp parallel if (parallel: Arg)
  fn3();
  return 0;
}

// CHECK-LABEL: define {{.*}}i{{[0-9]+}} @main()
int main() {
// CHECK: [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, void {{.+}}* [[CAP_FN4:@.+]] to void
#pragma omp parallel if (true)
  fn4();
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: store i32 [[GTID]], i32* [[GTID_ADDR:%.+]],
// CHECK: call void [[CAP_FN5:@.+]](i32* [[GTID_ADDR]],
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
#pragma omp parallel if (false)
  fn5();

// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, void {{.+}}* [[CAP_FN6:@.+]] to void
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: store i32 [[GTID]], i32* [[GTID_ADDR:%.+]],
// CHECK: call void [[CAP_FN6]](i32* [[GTID_ADDR]],
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]
#pragma omp parallel if (Arg)
  fn6();
  // CHECK: = call {{.*}}i{{.+}} @{{.+}}tmain
  return tmain(Arg);
}

// CHECK: define internal {{.*}}void [[CAP_FN4]]
// CHECK: call {{.*}}void @{{.+}}fn4
// CHECK: ret void

// CHECK: define internal {{.*}}void [[CAP_FN5]]
// CHECK: call {{.*}}void @{{.+}}fn5
// CHECK: ret void

// CHECK: define internal {{.*}}void [[CAP_FN6]]
// CHECK: call {{.*}}void @{{.+}}fn6
// CHECK: ret void

// CHECK-LABEL: define {{.+}} @{{.+}}tmain
// CHECK: [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, void {{.+}}* [[CAP_FN1:@.+]] to void
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: store i32 [[GTID]], i32* [[GTID_ADDR:%.+]],
// CHECK: call void [[CAP_FN2:@.+]](i32* [[GTID_ADDR]],
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call {{.*}}void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, void {{.+}}* [[CAP_FN3:@.+]] to void
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call {{.*}}void @__kmpc_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: store i32 [[GTID]], i32* [[GTID_ADDR:%.+]],
// CHECK: call void [[CAP_FN3]](i32* [[GTID_ADDR]],
// CHECK: call {{.*}}void @__kmpc_end_serialized_parallel(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]

// CHECK: define internal {{.*}}void [[CAP_FN1]]
// CHECK: call {{.*}}void @{{.+}}fn1
// CHECK: ret void

// CHECK: define internal {{.*}}void [[CAP_FN2]]
// CHECK: call {{.*}}void @{{.+}}fn2
// CHECK: ret void

// CHECK: define internal {{.*}}void [[CAP_FN3]]
// CHECK: call {{.*}}void @{{.+}}fn3
// CHECK: ret void

#endif
