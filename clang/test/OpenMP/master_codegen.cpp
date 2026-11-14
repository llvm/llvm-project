// RUN: %clang_cc1 -verify -fopenmp -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=ALL,NORMAL,OGCG
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,NORMAL,OGCG
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefixes=ALL,IRBUILDER,OGCG
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefixes=ALL,IRBUILDER,OGCG

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %if cir-enabled %{ %clang_cc1 -verify -fopenmp -std=c++11 -triple x86_64-unknown-linux-gnu -DCIR_TEST -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR %}
// RUN: %if cir-enabled %{ %clang_cc1 -verify -fopenmp -std=c++11 -triple x86_64-unknown-linux-gnu -DCIR_TEST -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,CIRLLVM %}

// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// ALL:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, ptr }

// ALL:       define {{.*}}void [[FOO:@.+]]()

void foo() { extern void mayThrow(); mayThrow(); }

// ALL-LABEL: @main
// CIR-LABEL: cir.func{{.*}} @main(
// TERM_DEBUG-LABEL: @main
int main() {
  // ALL:      			[[A_ADDR:%.+]] = alloca i8
  // CIR: [[A_ADDR:%.*]] = cir.alloca "a"
  char a;

// ALL:       			[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// ALL:       			[[RES:%.+]] = call {{.*}}i32 @__kmpc_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// ALL-NEXT:  			[[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// ALL-NEXT:  			br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// ALL:       			[[THEN]]
// CIRLLVM-NEXT: br label %[[MASTER_REGION:[^ ,]+]]
// CIRLLVM: [[MASTER_REGION]]:
// ALL-NEXT:  			store i8 2, ptr [[A_ADDR]]
// CIRLLVM-NEXT: br label %[[MASTER_CONT:[^ ,]+]]
// CIRLLVM: [[MASTER_CONT]]:
// CIRLLVM-NEXT: br label %[[MASTER_FINALIZE:[^ ,]+]]
// CIRLLVM: [[MASTER_FINALIZE]]:
// IRBUILDER-NEXT:		br label %[[AFTER:[^ ,]+]]
// IRBUILDER:			[[AFTER]]
// IRBUILDER-NEXT:		br label %[[OMP_REGION_FINALIZE:[^ ,]+]]
// IRBUILDER:			[[OMP_REGION_FINALIZE]]
// ALL-NEXT:  			call {{.*}}void @__kmpc_end_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// ALL-NEXT:  			br label {{%?}}[[EXIT]]
// ALL:       			[[EXIT]]
// CIR: omp.master {
// CIR-NEXT: [[TWO:%.*]] = cir.const #cir.int<2> : !s8i
// CIR-NEXT: cir.store align(1) [[TWO]], [[A_ADDR]]
// CIR-NEXT: omp.terminator
// CIR-NEXT: }
#pragma omp master
  a = 2;
// CIRLLVM: [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// IRBUILDER: 			[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@.+]])
// ALL:       			[[RES:%.+]] = call {{.*}}i32 @__kmpc_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// ALL-NEXT:  			[[IS_MASTER:%.+]] = icmp ne i32 [[RES]], 0
// ALL-NEXT:  			br i1 [[IS_MASTER]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// ALL:       			[[THEN]]
// CIRLLVM-NEXT: br label %[[MASTER_REGION:[^ ,]+]]
// CIRLLVM: [[MASTER_REGION]]:
// CIRLLVM-NEXT: call {{.*}}void [[FOO]]()
// CIRLLVM-NEXT: br label %[[MASTER_CONT:[^ ,]+]]
// CIRLLVM: [[MASTER_CONT]]:
// CIRLLVM-NEXT: br label %[[MASTER_FINALIZE:[^ ,]+]]
// CIRLLVM: [[MASTER_FINALIZE]]:
// IRBUILDER-NEXT:  call {{.*}}void [[FOO]]()
// NORMAL-NEXT:  		invoke {{.*}}void [[FOO]]()
// ALL:       			call {{.*}}void @__kmpc_end_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// ALL-NEXT:  			br label {{%?}}[[EXIT]]
// ALL:       			[[EXIT]]
// CIR: omp.master {
// CIR-NEXT: cir.call @_Z3foov(
// CIR-NEXT: omp.terminator
// CIR-NEXT: }
#pragma omp master
  foo();
  // ALL-NOT:   call i32 @__kmpc_master
  // ALL-NOT:   call void @__kmpc_end_master
  return a;
}

// ALL-LABEL:        lambda_master
// TERM_DEBUG-LABEL: lambda_master
void lambda_master(int a, int b) {
  auto l = [=]() {
#pragma omp master
    {
      // ALL: call i32 @__kmpc_master(
      // CIR-LABEL: cir.func{{.*}} @_ZZ13lambda_masteriiENK{{.*}}clEv(
      // CIR: omp.master {
      // CIR-NEXT: [[C_ADDR:%.*]] = cir.alloca "c"
      // CIR: [[SUM:%.*]] = cir.add nsw
      // CIR-NEXT: cir.store align(4) [[SUM]], [[C_ADDR]]
      // CIR-NEXT: omp.terminator
      // CIR-NEXT: }
      int c = a + b;
    }
  };

  l();

  auto l1 = [=]() {
#pragma omp parallel
#pragma omp master
    {
      // ALL: call i32 @__kmpc_master(
      // CIR-LABEL: cir.func{{.*}} @_ZZ13lambda_masteriiENK{{.*}}clEv(
      // CIR: omp.parallel {
      // CIR-NEXT: omp.master {
      // CIR-NEXT: [[C_ADDR:%.*]] = cir.alloca "c"
      // CIR: [[SUM:%.*]] = cir.add nsw
      // CIR-NEXT: cir.store align(4) [[SUM]], [[C_ADDR]]
      // CIR-NEXT: omp.terminator
      // CIR-NEXT: }
      // CIR-NEXT: omp.terminator
      // CIR-NEXT: }
      int c = a + b;
    }
  };

  l1();

#ifndef CIR_TEST
  auto l2 = [=]() {
#pragma omp parallel master
    {
      // OGCG: call i32 @__kmpc_master(
      int c = a + b;
    }
  };

  l2();
#endif
}

// ALL-LABEL:      parallel_master
// CIRLLVM: call {{.*}}@__kmpc_fork_call({{.*}}ptr [[OUTLINED:@[^, )]+]]
// CIRLLVM: define internal {{.*}}void [[OUTLINED]](
// CIRLLVM: [[GTID:%.*]] = call i32 @__kmpc_global_thread_num(ptr [[DEFAULT_LOC:@[^, )]+]])
// CIRLLVM-NEXT: [[RES:%.*]] = call i32 @__kmpc_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// CIRLLVM-NEXT: [[IS_MASTER:%.*]] = icmp ne i32 [[RES]], 0
// CIRLLVM-NEXT: br i1 [[IS_MASTER]], label %[[THEN:[^ ,]+]], label %[[EXIT:[^ ,]+]]
// CIRLLVM: [[THEN]]:
// CIRLLVM-NEXT: br label %[[MASTER_REGION:[^ ,]+]]
// CIRLLVM: [[MASTER_REGION]]:
// CIRLLVM-NEXT: call void [[FOO]]()
// CIRLLVM-NEXT: br label %[[MASTER_CONT:[^ ,]+]]
// CIRLLVM: [[MASTER_CONT]]:
// CIRLLVM-NEXT: br label %[[MASTER_FINALIZE:[^ ,]+]]
// CIRLLVM: [[MASTER_FINALIZE]]:
// CIRLLVM-NEXT: call void @__kmpc_end_master(ptr [[DEFAULT_LOC]], i32 [[GTID]])
// CIRLLVM-NEXT: br label %[[EXIT]]
// CIR-LABEL: cir.func{{.*}} @_Z15parallel_masterv(
// CIR: omp.parallel {
// CIR-NEXT: omp.master {
// CIR-NEXT: cir.call @_Z3foov(
// CIR-NEXT: omp.terminator
// CIR-NEXT: }
// CIR-NEXT: omp.terminator
// CIR-NEXT: }
// TERM_DEBUG-LABEL: parallel_master
void parallel_master() {
#pragma omp parallel
#pragma omp master
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call i32 @__kmpc_master({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_master({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();
}
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-12]],
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !DILocation(line: [[@LINE-3]],

#endif
