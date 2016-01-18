// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -fexceptions -fcxx-exceptions -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -include-pch %t -fsyntax-only -verify %s -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER
// CHECK: [[IMPLICIT_BARRIER_LOC:@.+]] = private unnamed_addr constant %{{.+}} { i32 0, i32 66, i32 0, i32 0, i8*
// CHECK-LABEL: foo
void foo() {};
// CHECK-LABEL: bar
void bar() {};

template <class T>
T tmain() {
#pragma omp parallel sections
  {
    foo();
  }
  return T();
}

// CHECK-LABEL: @main
int main() {
// CHECK: call void (%{{.+}}*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%{{.+}}* @{{.+}}, i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_PARALLEL_FUNC:@.+]] to void (i32*, i32*, ...)*))
// CHECK-LABEL: }
// CHECK: define internal void [[OMP_PARALLEL_FUNC]](i32* noalias [[GTID_PARAM_ADDR:%.+]], i32* noalias %{{.+}})
// CHECK: store i32* [[GTID_PARAM_ADDR]], i32** [[GTID_REF_ADDR:%.+]],
#pragma omp parallel sections
  {
// CHECK:      store i32 0, i32* [[LB_PTR:%.+]],
// CHECK:      store i32 1, i32* [[UB_PTR:%.+]],
// CHECK:      [[GTID_REF:%.+]] = load i32*, i32** [[GTID_REF_ADDR]],
// CHECK:      [[GTID:%.+]] = load i32, i32* [[GTID_REF]],
// CHECK:      call void @__kmpc_for_static_init_4(%{{.+}}* @{{.+}}, i32 [[GTID]], i32 34, i32* [[IS_LAST_PTR:%.+]], i32* [[LB_PTR]], i32* [[UB_PTR]], i32* [[STRIDE_PTR:%.+]], i32 1, i32 1)
// <<UB = min(UB, GlobalUB);>>
// CHECK:      [[UB:%.+]] = load i32, i32* [[UB_PTR]]
// CHECK:      [[CMP:%.+]] = icmp slt i32 [[UB]], 1
// CHECK:      [[MIN_UB_GLOBALUB:%.+]] = select i1 [[CMP]], i32 [[UB]], i32 1
// CHECK:      store i32 [[MIN_UB_GLOBALUB]], i32* [[UB_PTR]]
// <<IV = LB;>>
// CHECK:      [[LB:%.+]] = load i32, i32* [[LB_PTR]]
// CHECK:      store i32 [[LB]], i32* [[IV_PTR:%.+]]
// CHECK:      br label %[[INNER_FOR_COND:.+]]
// CHECK:      [[INNER_FOR_COND]]
// <<IV <= UB?>>
// CHECK:      [[IV:%.+]] = load i32, i32* [[IV_PTR]]
// CHECK:      [[UB:%.+]] = load i32, i32* [[UB_PTR]]
// CHECK:      [[CMP:%.+]] = icmp sle i32 [[IV]], [[UB]]
// CHECK:      br i1 [[CMP]], label %[[INNER_LOOP_BODY:.+]], label %[[INNER_LOOP_END:.+]]
// CHECK:      [[INNER_LOOP_BODY]]
// <<TRUE>> - > <BODY>
// CHECK:      [[IV:%.+]] = load i32, i32* [[IV_PTR]]
// CHECK:      switch i32 [[IV]], label %[[SECTIONS_EXIT:.+]] [
// CHECK-NEXT: i32 0, label %[[SECTIONS_CASE0:.+]]
// CHECK-NEXT: i32 1, label %[[SECTIONS_CASE1:.+]]
#pragma omp section
// CHECK:      [[SECTIONS_CASE0]]
// CHECK-NEXT: invoke void @{{.*}}foo{{.*}}()
// CHECK:      br label %[[SECTIONS_EXIT]]
    foo();
#pragma omp section
// CHECK:      [[SECTIONS_CASE1]]
// CHECK-NEXT: invoke void @{{.*}}bar{{.*}}()
// CHECK:      br label %[[SECTIONS_EXIT]]
    bar();
// CHECK:      [[SECTIONS_EXIT]]
// <<++IV;>>
// CHECK:      [[IV:%.+]] = load i32, i32* [[IV_PTR]]
// CHECK-NEXT: [[INC:%.+]] = add nsw i32 [[IV]], 1
// CHECK-NEXT: store i32 [[INC]], i32* [[IV_PTR]]
// CHECK-NEXT: br label %[[INNER_FOR_COND]]
// CHECK:      [[INNER_LOOP_END]]
  }
// CHECK:      call void @__kmpc_for_static_fini(%{{.+}}* @{{.+}}, i32 [[GTID]])
// CHECK:      call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]],
  return tmain<int>();
}

// CHECK-LABEL: tmain
// CHECK:       call void {{.*}} @__kmpc_fork_call(
// CHECK-NOT:   __kmpc_global_thread_num
// CHECK:       [[RES:%.+]] = call i32 @__kmpc_single(
// CHECK-NEXT:  [[BOOLRES:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  br i1 [[BOOLRES]], label %[[THEN:.+]], label %[[END:.+]]
// CHECK:       [[THEN]]
// CHECK-NEXT:  invoke void @{{.*}}foo{{.*}}()
// CHECK-NEXT:  unwind label %[[TERM_LPAD:.+]]
// CHECK:       call void @__kmpc_end_single(
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  call void @__kmpc_barrier(%{{.+}}* [[IMPLICIT_BARRIER_LOC]],
// CHECK-NEXT:  ret
// CHECK:       [[TERM_LPAD]]
// CHECK:       call void @__clang_call_terminate(i8*
// CHECK-NEXT:  unreachable

#endif
