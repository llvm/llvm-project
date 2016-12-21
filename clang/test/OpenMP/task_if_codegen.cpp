// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple x86_64-apple-darwin10 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void fn1();
void fn2();
void fn3();
void fn4();
void fn5();
void fn6();
void fn7();
void fn8();
void fn9();
void fn10();

int Arg;

// CHECK-LABEL: define void @{{.+}}gtid_test
void gtid_test() {
// CHECK:  call void {{.+}} @__kmpc_fork_call(%{{.+}}* @{{.+}}, i{{.+}} 0, {{.+}}* [[GTID_TEST_REGION1:@.+]] to void
#pragma omp parallel
#pragma omp task if (task: false)
  gtid_test();
// CHECK: ret void
}

// CHECK: define internal void [[GTID_TEST_REGION1]](i32* noalias [[GTID_PARAM:%.+]], i
// CHECK: store i32* [[GTID_PARAM]], i32** [[GTID_ADDR_REF:%.+]],
// CHECK: [[GTID_ADDR:%.+]] = load i32*, i32** [[GTID_ADDR_REF]]
// CHECK: [[GTID:%.+]] = load i32, i32* [[GTID_ADDR]]
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[GTID_TEST_REGION2:@.+]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: ret void

// CHECK: define internal i32 [[GTID_TEST_REGION2]](
// CHECK: call void @{{.+}}gtid_test
// CHECK: ret i32

template <typename T>
int tmain(T Arg) {
#pragma omp task if (task: true)
  fn1();
#pragma omp task if (false)
  fn2();
#pragma omp task if (Arg)
  fn3();
#pragma omp task if (task: Arg) depend(in : Arg)
  fn4();
#pragma omp task if (Arg) depend(out : Arg)
  fn5();
#pragma omp task if (Arg) depend(inout : Arg)
  fn6();
  return 0;
}

// CHECK-LABEL: @main
int main() {
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc({{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN7:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]])
#pragma omp task if (true)
  fn7();
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc({{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN8:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN8]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
#pragma omp task if (false)
  fn8();

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc({{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN9:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN9]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]
#pragma omp task if (Arg)
  fn9();
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc({{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN10:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task_with_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]], i32 1, i8* [[LIST:%[^,]+]], i32 0, i8* null)
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_wait_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i8* [[LIST]], i32 0, i8* null)
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN10]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]
#pragma omp task if (Arg) depend(inout : Arg)
  fn10();
  // CHECK: = call {{.*}}i{{.+}} @{{.+}}tmain
  return tmain(Arg);
}

// CHECK: define internal i32 [[CAP_FN7]]
// CHECK: call void @{{.+}}fn7
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN8]]
// CHECK: call void @{{.+}}fn8
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN9]]
// CHECK: call void @{{.+}}fn9
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN10]]
// CHECK: call void @{{.+}}fn10
// CHECK: ret i32

// CHECK-LABEL: define {{.+}} @{{.+}}tmain
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32,  %{{[^*]+}}*)* [[CAP_FN1:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]])

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN2:@.+]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN3:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN3]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN4:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task_with_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]], i32 1, i8* [[LIST:%.+]], i32 0, i8* null)
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_wait_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i8* [[LIST]], i32 0, i8* null)
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN4]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN5:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task_with_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]], i32 1, i8* [[LIST:%.+]], i32 0, i8* null)
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_wait_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i8* [[LIST]], i32 0, i8* null)
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN5]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]

// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%{{[^,]+}}, i32 [[GTID]], i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, %{{[^*]+}}*)* [[CAP_FN6:[^ ]+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_PTR:%.+]] = bitcast i8* [[ORIG_TASK_PTR]] to
// CHECK: br i1 %{{.+}}, label %[[OMP_THEN:.+]], label %[[OMP_ELSE:.+]]
// CHECK: [[OMP_THEN]]
// CHECK: call i32 @__kmpc_omp_task_with_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i8* [[ORIG_TASK_PTR]], i32 1, i8* [[LIST:%.+]], i32 0, i8* null)
// CHECK: br label %[[OMP_END:.+]]
// CHECK: [[OMP_ELSE]]
// CHECK: call void @__kmpc_omp_wait_deps(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 1, i8* [[LIST]], i32 0, i8* null)
// CHECK: call void @__kmpc_omp_task_begin_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: call i32 [[CAP_FN6]](i32 [[GTID]], %{{.+}}* [[TASK_PTR]])
// CHECK: call void @__kmpc_omp_task_complete_if0(%{{.+}}* @{{.+}}, i{{.+}} [[GTID]], i8* [[ORIG_TASK_PTR]])
// CHECK: br label %[[OMP_END]]
// CHECK: [[OMP_END]]

// CHECK: define internal i32 [[CAP_FN1]]
// CHECK: call void @{{.+}}fn1
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN2]]
// CHECK: call void @{{.+}}fn2
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN3]]
// CHECK: call void @{{.+}}fn3
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN4]]
// CHECK: call void @{{.+}}fn4
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN5]]
// CHECK: call void @{{.+}}fn5
// CHECK: ret i32

// CHECK: define internal i32 [[CAP_FN6]]
// CHECK: call void @{{.+}}fn6
// CHECK: ret i32

#endif
