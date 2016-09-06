// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// expected-no-diagnostics
// REQUIRES: x86-registered-target
#ifndef HEADER
#define HEADER

// CHECK:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK:       [[UNNAMED_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// CHECK:       [[THE_NAME_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// CHECK:       [[THE_NAME_LOCK1:@.+]] = common global [8 x i32] zeroinitializer

// CHECK:       define {{.*}}void [[FOO:@.+]]()

void foo() {}

// CHECK-LABEL: @main
// TERM_DEBUG-LABEL: @main
int main() {
// CHECK:       [[A_ADDR:%.+]] = alloca i8
  char a;

// CHECK:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       call {{.*}}void @__kmpc_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[UNNAMED_LOCK]])
// CHECK-NEXT:  store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  call {{.*}}void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[UNNAMED_LOCK]])
#pragma omp critical
  a = 2;
// CHECK:       call {{.*}}void @__kmpc_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
// CHECK-NEXT:  invoke {{.*}}void [[FOO]]()
// CHECK:       call {{.*}}void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
#pragma omp critical(the_name)
  foo();
// CHECK:       call {{.*}}void @__kmpc_critical_with_hint([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK1]], i{{64|32}} 23)
// CHECK-NEXT:  invoke {{.*}}void [[FOO]]()
// CHECK:       call {{.*}}void @__kmpc_end_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK1]])
#pragma omp critical(the_name1) hint(23)
  foo();
// CHECK:       call {{.*}}void @__kmpc_critical([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], [8 x i32]* [[THE_NAME_LOCK]])
// CHECK:       br label
// CHECK-NOT:   call {{.*}}void @__kmpc_end_critical(
// CHECK:       br label
// CHECK-NOT:   call {{.*}}void @__kmpc_end_critical(
// CHECK:       br label
  if (a)
#pragma omp critical(the_name)
    while (1)
      ;
// CHECK:  call {{.*}}void [[FOO]]()
  foo();
// CHECK-NOT:   call void @__kmpc_critical
// CHECK-NOT:   call void @__kmpc_end_critical
  return a;
}

struct S {
  int a;
};
// CHECK-LABEL: critical_ref
void critical_ref(S &s) {
  // CHECK: [[S_ADDR:%.+]] = alloca %struct.S*,
  // CHECK: [[S_REF:%.+]] = load %struct.S*, %struct.S** [[S_ADDR]],
  // CHECK: [[S_A_REF:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_REF]], i32 0, i32 0
  ++s.a;
  // CHECK: [[S_REF:%.+]] = load %struct.S*, %struct.S** [[S_ADDR]],
  // CHECK: store %struct.S* [[S_REF]], %struct.S** [[S_ADDR:%.+]],
  // CHECK: call void @__kmpc_critical(
#pragma omp critical
  // CHECK: [[S_REF:%.+]] = load %struct.S*, %struct.S** [[S_ADDR]],
  // CHECK: [[S_A_REF:%.+]] = getelementptr inbounds %struct.S, %struct.S* [[S_REF]], i32 0, i32 0
  ++s.a;
  // CHECK: call void @__kmpc_end_critical(
}

// CHECK-LABEL:      parallel_critical
// TERM_DEBUG-LABEL: parallel_critical
void parallel_critical() {
#pragma omp parallel
#pragma omp critical
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_critical({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_critical({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();
}
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-12]],
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !DILocation(line: [[@LINE-3]],
#endif
