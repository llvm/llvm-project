// RUN: %clang_cc1 -verify -fopenmp -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=ALL,NORMAL
// RUN: %clang_cc1 -fopenmp -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,NORMAL
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=ALL,IRBUILDER
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,IRBUILDER

// RUN: %clang_cc1 -verify -fopenmp-simd -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// ALL:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, ptr }
// ALL:       [[UNNAMED_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// ALL:       [[THE_NAME_LOCK:@.+]] = common global [8 x i32] zeroinitializer
// ALL:       [[THE_NAME_LOCK1:@.+]] = common global [8 x i32] zeroinitializer

// ALL:       define {{.*}}void [[FOO:@.+]]()

void foo() { extern void mayThrow(); mayThrow(); }

// ALL-LABEL: @main
// TERM_DEBUG-LABEL: @main
int main() {
  // ALL:       [[A_ADDR:%.+]] = alloca i8
  char a;

// ALL:       			[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// ALL:       			call {{.*}}void @__kmpc_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[UNNAMED_LOCK]])
// ALL-NEXT:  			store i8 2, ptr [[A_ADDR]]
// IRBUILDER-NEXT:		br label %[[AFTER:[^ ,]+]]
// IRBUILDER:			[[AFTER]]
// ALL-NEXT:  			call {{.*}}void @__kmpc_end_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[UNNAMED_LOCK]])
  [[omp::directive(critical)]]
  a = 2;
// IRBUILDER:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// ALL:       			call {{.*}}void @__kmpc_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[THE_NAME_LOCK]])
// IRBUILDER-NEXT:	call {{.*}}void [[FOO]]()
// NORMAL-NEXT:  		invoke {{.*}}void [[FOO]]()
// IRBUILDER-NEXT:		br label %[[AFTER:[^ ,]+]]
// IRBUILDER:			[[AFTER]]
// ALL:      				call {{.*}}void @__kmpc_end_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[THE_NAME_LOCK]])
  [[omp::directive(critical(the_name))]]
  foo();
// IRBUILDER:   		[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// ALL: 	      		call {{.*}}void @__kmpc_critical_with_hint(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[THE_NAME_LOCK1]], i{{64|32}} 23)
// IRBUILDER-NEXT:	call {{.*}}void [[FOO]]()
// NORMAL-NEXT:		  invoke {{.*}}void [[FOO]]()
// IRBUILDER-NEXT:		br label %[[AFTER:[^ ,]+]]
// IRBUILDER:			[[AFTER]]
// ALL:		       		call {{.*}}void @__kmpc_end_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[THE_NAME_LOCK1]])
  [[omp::directive(critical(the_name1) hint(23))]]
  foo();
  // IRBUILDER:   		[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
  // ALL:       call {{.*}}void @__kmpc_critical(ptr [[DEFAULT_LOC]], i32 [[GTID]], ptr [[THE_NAME_LOCK]])
  // NORMAL:       br label
  // NORMAL-NOT:   call {{.*}}void @__kmpc_end_critical(
  // NORMAL:       br label
  // NORMAL-NOT:   call {{.*}}void @__kmpc_end_critical(
  // NORMAL:       br label
  if (a)
    [[omp::directive(critical(the_name))]]
    while (1)
      ;
  // ALL:  call {{.*}}void [[FOO]]()
  foo();
  // ALL-NOT:   call void @__kmpc_critical
  // ALL-NOT:   call void @__kmpc_end_critical
  return a;
}

// ALL-LABEL:        lambda_critical
// TERM_DEBUG-LABEL: lambda_critical
void lambda_critical(int a, int b) {
  auto l = [=]() {
    [[omp::directive(critical)]]
    {
      // ALL: call void @__kmpc_critical(
      int c = a + b;
    }
  };

  l();

  auto l1 = [=]() {
    [[omp::sequence(directive(parallel), directive(critical))]]
    {
      // ALL: call void @__kmpc_critical(
      int c = a + b;
    }
  };

  l1();
}

struct S {
  int a;
};
// ALL-LABEL: critical_ref
void critical_ref(S &s) {
  // ALL: [[S_ADDR:%.+]] = alloca ptr,
  // ALL: [[S_REF:%.+]] = load ptr, ptr [[S_ADDR]],
  // ALL: [[S_A_REF:%.+]] = getelementptr inbounds %struct.S, ptr [[S_REF]], i32 0, i32 0
  ++s.a;
  // ALL: call void @__kmpc_critical(
  [[omp::directive(critical)]]
  // ALL: [[S_REF:%.+]] = load ptr, ptr [[S_ADDR]],
  // ALL: [[S_A_REF:%.+]] = getelementptr inbounds %struct.S, ptr [[S_REF]], i32 0, i32 0
  ++s.a;
  // ALL: call void @__kmpc_end_critical(
}

// ALL-LABEL:      parallel_critical
// TERM_DEBUG-LABEL: parallel_critical
void parallel_critical() {
  [[omp::sequence(directive(parallel), directive(critical))]]
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

