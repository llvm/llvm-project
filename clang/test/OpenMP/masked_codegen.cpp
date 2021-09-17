// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=51 -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck %s --check-prefix=TERM_DEBUG

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=51 -fexceptions -fcxx-exceptions -debug-info-kind=line-tables-only -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK:       [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }

// CHECK:       define {{.*}}void [[FOO:@.+]]()

void foo() { extern void mayThrow(); mayThrow(); }

// CHECK-LABEL: @main
// TERM_DEBUG-LABEL: @main
int main() {
  // CHECK:      			[[A_ADDR:%.+]] = alloca i8
  char a;

// CHECK:       			[[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       			[[RES:%.+]] = call {{.*}}i32 @__kmpc_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 0)
// CHECK-NEXT:  			[[IS_MASKED:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  			br i1 [[IS_MASKED]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       			[[THEN]]
// CHECK-NEXT:  			store i8 2, i8* [[A_ADDR]]
// CHECK-NEXT:  			call {{.*}}void @__kmpc_end_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  			br label {{%?}}[[EXIT]]
// CHECK:       			[[EXIT]]
#pragma omp masked
  a = 2;
// CHECK: 			        [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       			[[RES:%.+]] = call {{.*}}i32 @__kmpc_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 2)
// CHECK-NEXT:  			[[IS_MASKED:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  			br i1 [[IS_MASKED]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       			[[THEN]]
// CHECK-NEXT:                          call {{.*}}void [[FOO]]()
// CHECK:       			call {{.*}}void @__kmpc_end_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  			br label {{%?}}[[EXIT]]
// CHECK:       			[[EXIT]]
#pragma omp masked filter(2)
  foo();
// CHECK:                               store i32 9, i32* [[X:.+]],
// CHECK:                               [[X_VAL:%.+]] = load i32, i32* [[X]]
// CHECK: 			        [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEFAULT_LOC:@.+]])
// CHECK:       			[[RES:%.+]] = call {{.*}}i32 @__kmpc_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]], i32 [[X_VAL]])
// CHECK-NEXT:  			[[IS_MASKED:%.+]] = icmp ne i32 [[RES]], 0
// CHECK-NEXT:  			br i1 [[IS_MASKED]], label {{%?}}[[THEN:.+]], label {{%?}}[[EXIT:.+]]
// CHECK:       			[[THEN]]
// CHECK-NEXT:                          call {{.*}}void [[FOO]]()
// CHECK:       			call {{.*}}void @__kmpc_end_masked([[IDENT_T_TY]]* [[DEFAULT_LOC]], i32 [[GTID]])
// CHECK-NEXT:  			br label {{%?}}[[EXIT]]
// CHECK:       			[[EXIT]]
  int x = 9;
#pragma omp masked filter(x)
  foo();
  // CHECK-NOT:   call i32 @__kmpc_masked
  // CHECK-NOT:   call void @__kmpc_end_masked
  return a;
}

// CHECK-LABEL:        lambda_masked
// TERM_DEBUG-LABEL: lambda_masked
void lambda_masked(int a, int b) {
  auto l = [=]() {
#pragma omp masked
    {
      // CHECK: call i32 @__kmpc_masked(
      int c = a + b;
    }
  };

  l();

  auto l1 = [=]() {
#pragma omp parallel
#pragma omp masked filter(1)
    {
      // CHECK: call i32 @__kmpc_masked(
      int c = a + b;
    }
  };

  l1();

  int y = 1;
  auto l2 = [=](int yy) {
#pragma omp parallel
#pragma omp masked filter(yy)
    {
      // CHECK: call i32 @__kmpc_masked(
      int c = a + b;
    }
  };

  l2(y);
}

// CHECK-LABEL:      parallel_masked
// TERM_DEBUG-LABEL: parallel_masked
void parallel_masked() {
#pragma omp parallel
#pragma omp masked filter(1)
  // TERM_DEBUG: __kmpc_global_thread_num
  // TERM_DEBUG:     call i32 @__kmpc_masked({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_masked({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();

  int x;
#pragma omp parallel
#pragma omp masked filter(x)
  // TERM_DEBUG: __kmpc_global_thread_num
  // TERM_DEBUG:     call i32 @__kmpc_masked({{.+}}), !dbg [[DBG_LOC_START:![0-9]+]]
  // TERM_DEBUG:     invoke void {{.*}}foo{{.*}}()
  // TERM_DEBUG:     unwind label %[[TERM_LPAD:.+]],
  // TERM_DEBUG-NOT: __kmpc_global_thread_num
  // TERM_DEBUG:     call void @__kmpc_end_masked({{.+}}), !dbg [[DBG_LOC_END:![0-9]+]]
  // TERM_DEBUG:     [[TERM_LPAD]]
  // TERM_DEBUG:     call void @__clang_call_terminate
  // TERM_DEBUG:     unreachable
  foo();
}
// TERM_DEBUG-DAG: [[DBG_LOC_START]] = !DILocation(line: [[@LINE-12]],
// TERM_DEBUG-DAG: [[DBG_LOC_END]] = !DILocation(line: [[@LINE-3]],

#endif
