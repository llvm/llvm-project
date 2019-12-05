// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#ifdef CK1
///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK1
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK1

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// CK1-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK1-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK1-DAG: [[DEF_LOC:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

// CK1-LABEL: foo
void foo() {}

void parallel_master() {
#pragma omp parallel master
  foo();
}

// CK1-LABEL: define void @{{.+}}parallel_master{{.+}}
// CK1:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*))

// CK1:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GTID:%.+]], i32* noalias [[BTID:%.+]])
// CK1-NOT:   __kmpc_global_thread_num
// CK1:       call i32 @__kmpc_master({{.+}})
// CK1:       invoke void {{.*}}foo{{.*}}()
// CK1-NOT:   __kmpc_global_thread_num
// CK1:       call void @__kmpc_end_master({{.+}})
// CK1:       call void @__clang_call_terminate
// CK1:       unreachable

#endif

#ifdef CK2
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK2
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK2

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// CK2-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK2-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK2-DAG: [[DEF_LOC:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void parallel_master_private() {
  int a;
#pragma omp parallel master private(a)
  a++;
}

// CK2-LABEL: define void @{{.+}}parallel_master_private{{.+}}
// CK2:       [[A_PRIV:%.+]] = alloca i32
// CK2:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*))

// CK2:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GTID:%.+]], i32* noalias [[BTID:%.+]])
// CK2-NOT:   __kmpc_global_thread_num
// CK2:       call i32 @__kmpc_master({{.+}})
// CK2:       [[A_VAL:%.+]] = load i32, i32* [[A_PRIV]]
// CK2:       [[INC:%.+]] = add nsw i32 [[A_VAL]]
// CK2:       store i32 [[INC]], i32* [[A_PRIV]]
// CK2-NOT:   __kmpc_global_thread_num
// CK2:       call void @__kmpc_end_master({{.+}})
// CK2:       ret void

#endif

#ifdef CK3
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK3
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK3

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// CK3-DAG:   %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK3-DAG:   [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"

void parallel_master_private() {
  int a;
#pragma omp parallel master default(shared)
  a++;
}

// CK3-LABEL: define void @{{.+}}parallel_master{{.+}}
// CK3:       [[A_VAL:%.+]] = alloca i32
// CK3:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* {{.+}}, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* [[OMP_OUTLINED:@.+]] to void 

// CK3:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GTID:%.+]], i32* noalias [[BTID:%.+]], i32* dereferenceable(4) [[A_VAL]])
// CK3:       [[GTID_ADDR:%.+]] = alloca i32*
// CK3:       [[BTID_ADDR:%.+]] = alloca i32*
// CK3:       [[A_ADDR:%.+]] = alloca i32* 
// CK3:       store i32* [[GTID]], i32** [[GTID_ADDR]]
// CK3:       store i32* [[BTID]], i32** [[BTID_ADDR]]
// CK3:       store i32* [[A_VAL]], i32** [[A_ADDR]]
// CK3:       [[ZERO:%.+]] = load i32*, i32** [[A_ADDR]]
// CK3-NOT:   __kmpc_global_thread_num
// CK3:       call i32 @__kmpc_master({{.+}})
// CK3:       [[FIVE:%.+]] = load i32, i32* [[ZERO]]
// CK3:       [[INC:%.+]] = add nsw i32 [[FIVE]]
// CK3:       store i32 [[INC]], i32* [[ZERO]]
// CK3-NOT:   __kmpc_global_thread_num
// CK3:       call void @__kmpc_end_master({{.+}})

#endif

#ifdef CK4
///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK4
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK4

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// CK4-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK4-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK4-DAG: [[DEF_LOC:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void parallel_master_firstprivate() {
  int a;
#pragma omp parallel master firstprivate(a)
  a++;
}

// CK4-LABEL: define void @{{.+}}parallel_master_firstprivate{{.+}}
// CK4:       [[A_VAL:%.+]] = alloca i32
// CK4:       [[A_CASTED:%.+]] = alloca i64
// CK4:       [[ZERO:%.+]] = load i32, i32* [[A_VAL]]
// CK4:       [[CONV:%.+]] = bitcast i64* [[A_CASTED]] to i32*
// CK4:       store i32 [[ZERO]], i32* [[CONV]]
// CK4:       [[ONE:%.+]] = load i64, i64* [[A_CASTED]]
// CK4:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i64 [[ONE]])

// CK4:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GLOBAL_TID:%.+]], i32* noalias [[BOUND_TID:%.+]], i64 [[A_VAL]])
// CK4:       [[GLOBAL_TID_ADDR:%.+]] = alloca i32*
// CK4:       [[BOUND_TID_ADDR:%.+]] = alloca i32*
// CK4:       [[A_ADDR:%.+]] = alloca i64 
// CK4:       store i32* [[GLOBAL_TID]], i32** [[GLOBAL_TID_ADDR]]
// CK4:       store i32* [[BOUND_TID]], i32** [[BOUND_TID_ADDR]]
// CK4:       store i64 [[A_VAL]], i64* [[A_ADDR]]
// CK4:       [[CONV]] = bitcast i64* [[A_ADDR]] to i32*
// CK4-NOT:   __kmpc_global_thread_num
// CK4:       call i32 @__kmpc_master({{.+}})
// CK4:       [[FOUR:%.+]] = load i32, i32* [[CONV]]
// CK4:       [[INC:%.+]] = add nsw i32 [[FOUR]]
// CK4:       store i32 [[INC]], i32* [[CONV]]
// CK4-NOT:   __kmpc_global_thread_num
// CK4:       call void @__kmpc_end_master({{.+}})

#endif

#ifdef CK5
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK5
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK5

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fnoopenmp-use-tls -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fnoopenmp-use-tls -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=TLS-CHECK
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TLS-CHECK

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple x86_64-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// CK5-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK5-DAG: [[A:@.+]] = {{.+}} i32 0
// CK5-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK5-DAG: [[DEF_LOC_1:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CK5-DAG: [[A_CACHE:@.+]] = common global i8** null
// CK5-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// TLS-CHECK-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// TLS-CHECK-DAG: [[A:@.+]] = thread_local global i32 0
// TLS-CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// TLS-CHECK-DAG: [[DEF_LOC_1:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 66, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// TLS-CHECK-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

int a;
#pragma omp threadprivate(a)

void parallel_master_copyin() {
#pragma omp parallel master copyin(a)
  a++;
}

// CK5-LABEL: define void @{{.+}}parallel_master_copyin{{.+}}
// CK5:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC_1]], i32 0, void (i32*, i32*, ...)* bitcast (void (i32*, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*))
// CK5: ret void
// TLS-CHECK-LABEL: define void @{{.+}}parallel_master_copyin{{.+}}
// TLS-CHECK:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC_2]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*)
// TLS-CHECK: ret void

// CK5:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GLOBAL_TID:%.+]], i32* noalias [[BOUND_TID:%.+]])
// CK5:       [[GLOBAL_TID_ADDR:%.+]] = alloca i32*
// CK5:       [[BOUND_TID_ADDR:%.+]] = alloca i32*
// CK5:       store i32* [[GLOBAL_TID]], i32** [[GLOBAL_TID_ADDR]]
// CK5:       store i32* [[BOUND_TID]], i32** [[BOUND_TID_ADDR]]
// CK5:       [[ZERO:%.+]] = load i32*, i32** [[GLOBAL_TID_ADDR]]
// CK5:       [[ONE:%.+]] = load i32, i32* [[ZERO]]
// CK5:       [[TWO:%.+]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* [[DEF_LOC_1]], i32 [[ONE]], i8* bitcast (i32* [[A]] to i8*), i64 4, i8*** [[A_CACHE]])
// CK5:       [[THREE:%.+]] = bitcast i8* [[TWO]] to i32*
// CK5:       [[FOUR:%.+]] = ptrtoint i32* [[THREE]] to i64
// CK5:       [[FIVE:%.+]] = icmp ne i64 ptrtoint (i32* [[A]] to i64), [[FOUR]]
// CK5:       br i1 [[FIVE]], label [[COPYIN_NOT_MASTER:%.+]], label [[COPYIN_NOT_MASTER_END:%.+]]
// TLS-CHECK: define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GLOBAL_TID:%.+]], i32* noalias [[BOUND_TID:%.+]], i32* {{.+}} [[A_VAR:%.+]])
// TLS-CHECK: [[GLOBAL_TID_ADDR:%.+]] = alloca i32*
// TLS-CHECK: [[BOUND_TID_ADDR:%.+]] = alloca i32*
// TLS-CHECK: [[A_ADDR:%.+]] = alloca i32*
// TLS-CHECK: store i32* [[A_VAR]], i32** [[A_ADDR]]
// TLS-CHECK: [[ZERO:%.+]] = load i32*, i32** [[A_ADDR]]
// TLS-CHECK: [[ONE:%.+]] = ptrtoint i32* [[ZERO]] to i64
// TLS-CHECK: [[TWO:%.+]] = icmp ne i64 [[ONE]], ptrtoint (i32* [[A]] to i64)
// TLS-CHECK: br i1 [[TWO]], label [[COPYIN_NOT_MASTER:%.+]], label [[COPYIN_NOT_MASTER_END:%.+]]

// CK5-DAG:   [[COPYIN_NOT_MASTER]]
// CK5-DAG:   [[SIX:%.+]] = load i32, i32* [[A]]
// TLS-CHECK-DAG:   [[COPYIN_NOT_MASTER]]
// TLS-CHECK-DAG:   [[THREE:%.+]] = load i32, i32* [[ZERO]]
// TLS-CHECK-DAG:   store i32 [[THREE]], i32* [[A]]

// CK5-DAG:   [[COPYIN_NOT_MASTER_END]]
// CK5-DAG:   call void @__kmpc_barrier(%struct.ident_t* [[DEF_LOC_2]], i32 [[ONE]])
// CK5-DAG:   [[SEVEN:%.+]] = call i32 @__kmpc_master(%struct.ident_t* [[DEF_LOC_1]], i32 [[ONE]])
// CK5-DAG:   [[EIGHT:%.+]] = icmp ne i32 [[SEVEN]], 0
// CK5-DAG:   br i1 %8, label [[OMP_IF_THEN:%.+]], label [[OMP_IF_END:%.+]]
// TLS-CHECK-DAG: [[FOUR:%.+]] = load i32*, i32** [[GLOBAL_TID_ADDR:%.+]]
// TLS-CHECK-DAG: [[FIVE:%.+]] = load i32, i32* [[FOUR]]
// TLS-CHECK-DAG: call void @__kmpc_barrier(%struct.ident_t* [[DEF_LOC_1]], i32 [[FIVE]])
// TLS-CHECK-DAG: [[SIX:%.+]] = load i32*, i32** [[GLOBAL_TID_ADDR]]
// TLS-CHECK-DAG: [[SEVEN:%.+]] = load i32, i32* [[SIX]]
// TLS-CHECK-DAG: [[EIGHT:%.+]] = call i32 @__kmpc_master(%struct.ident_t* [[DEF_LOC_2]], i32 [[SEVEN]])
// TLS-CHECK-DAG: [[NINE:%.+]] = icmp ne i32 [[EIGHT]], 0
// TLS-CHECK-DAG: br i1 [[NINE]], label [[OMP_IF_THEN:%.+]], label [[OMP_IF_END:%.+]]

// CK5-DAG:   [[OMP_IF_THEN]]
// CK5-DAG:   [[NINE:%.+]] = call i8* @__kmpc_threadprivate_cached(%struct.ident_t* [[DEF_LOC_1]], i32 %1, i8* bitcast (i32* [[A]] to i8*), i64 4, i8*** [[A_CACHE]])
// CK5-DAG:   [[TEN:%.+]] = bitcast i8* [[NINE]] to i32*
// CK5-DAG:   [[ELEVEN:%.+]] = load i32, i32* [[TEN]]
// CK5-DAG:   [[INC:%.+]] = add nsw i32 [[ELEVEN]], 1
// CK5-DAG:   store i32 [[INC]], i32* [[TEN]]
// CK5-DAG:   call void @__kmpc_end_master(%struct.ident_t* [[DEF_LOC_1]], i32 [[ONE]])
// CK5-DAG:   [[OMP_IF_END]]
// CK5-DAG:   ret void

// TLS-CHECK-DAG:   [[OMP_IF_THEN]]
// TLS-CHECK-DAG:   [[TEN:%.+]] = load i32, i32* [[A]]
// TLC-CHECK-DAG:   [[INC:%.+]] = add nsw i32 [[TEN]], 1
// TLC-CHECK-DAG:   store i32 [[INC]], i32* [[TEN]]
// TLS-CHECK-DAG:   call void @__kmpc_end_master(%struct.ident_t* [[DEF_LOC_2]], i32 [[SEVEN]])
// TLS-CHECK-DAG:   [[OMP_IF_END]]
// TLS-CHECK-DAG:   ret void

#endif
#ifdef CK6
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CK6
// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=CK6

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// CK6-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK6-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK6-DAG: [[DEF_LOC_1:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }
// CK6-DAG: [[GOMP:@.+]] = common global [8 x i32] zeroinitializer
// CK6-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 18, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void parallel_master_reduction() {
  int g;
#pragma omp parallel master reduction(+:g)
  g = 1;
}

// CK6-LABEL: define void @{{.+}}parallel_master_reduction{{.+}}
// CK6:       [[G_VAR:%.+]] = alloca i32
// CK6:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC_1]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i32* [[G_VAR]])
// CK6: ret void

// CK6:       define internal void [[OMP_OUTLINED]](i32* noalias [[GLOBAL_TID:%.+]], i32* noalias [[BOUND_TID:%.+]], i32* {{.+}} [[G_VAR]])
// CK6:       [[GTID_ADDR:%.+]] = alloca i32*
// CK6:       [[BTID_ADDR:%.+]] = alloca i32*
// CK6:       [[G_ADDR:%.+]] = alloca i32*
// CK6:       [[G_1:%.+]] = alloca i32
// CK6:       [[RED_LIST:%.+]] = alloca [1 x i8*]
// CK6:       [[ZERO:%.+]] = load i32*, i32** [[G_ADDR]]
// CK6:       [[ONE:%.+]] = load i32*, i32** [[GTID_ADDR]]
// CK6:       [[TWO:%.+]] = load i32, i32* [[ONE]]
// CK6:       [[THREE:%.+]] = call i32 @__kmpc_master(%struct.ident_t* [[DEF_LOC_1]], i32 [[TWO]])
// CK6:       [[FOUR:%.+]] = icmp ne i32 [[THREE]]

// CK6:       store i32 1, i32* [[G_1]]
// CK6:       call void @__kmpc_end_master(%struct.ident_t* [[DEF_LOC_1]], i32 [[TWO]])

// CK6:       [[FIVE:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[RED_LIST]], i64 0, i64 0
// CK6:       [[SEVEN:%.+]] = bitcast [1 x i8*]* [[RED_LIST]] to i8*
// CK6:       [[EIGHT:%.+]] = call i32 @__kmpc_reduce_nowait(%struct.ident_t* [[DEF_LOC_2]], i32 [[TWO]], i32 1, i64 8, i8* [[SEVEN]], void (i8*, i8*)* [[RED_FUNC:@.+]], [8 x i32]* [[RED_VAR:@.+]])

// switch
// CK6:       switch i32 [[EIGHT]], label [[RED_DEFAULT:%.+]] [
// CK6:       i32 1, label [[CASE1:%.+]]
// CK6:       i32 2, label [[CASE2:%.+]]

// case 1:
// CK6:       [[NINE:%.+]] = load i32, i32* %0, align 4
// CK6:       [[TEN:%.+]] = load i32, i32* [[G_1]]
// CK6:       [[ADD:%.+]] = add nsw i32 [[NINE]], [[TEN]]
// CK6:       store i32 [[ADD]], i32* [[ZERO]]
// CK6:       call void @__kmpc_end_reduce_nowait(%struct.ident_t* [[DEF_LOC_2]], i32 [[TWO]], [8 x i32]* [[GOMP]])
// CK6:       br label [[RED_DEFAULT]]

// case 2:
// CK6:       [[ELEVEN:%.+]] = load i32, i32* [[G_1]]
// CK6:       [[TWELVE:%.+]] = atomicrmw add i32* [[ZERO]], i32 [[ELEVEN]] monotonic

// CK6:       define internal void [[RED_FUNC]](i8* [[ZERO]], i8* [[ONE]])
// CK6:       ret void
#endif
#ifdef CK7
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK7
// RUN: %clang_cc1 -DCK7 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK7 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK7

// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// CK7-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK7-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK7-DAG: [[DEF_LOC_1:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void parallel_master_if() {
#pragma omp parallel master if (parallel: false)
  parallel_master_if();
}

// CK7-LABEL: parallel_master_if
// CK7:       [[ZERO:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t* [[DEF_LOC_1]])
// CK7:       call void @__kmpc_serialized_parallel(%struct.ident_t* [[DEF_LOC_1]], i32 [[ZERO]])
// CK7:       call void [[OUTLINED:@.+]](i32* [[THREAD_TEMP:%.+]], i32* [[BND_ADDR:%.+]])
// CK7:       call void @__kmpc_end_serialized_parallel(%struct.ident_t* [[DEF_LOC_1]], i32 [[ZERO]])
// CK7:       ret void

// CK7:       define internal void @.omp_outlined.(i32* noalias [[GTID:%.+]], i32* noalias [[BTID:%.+]])
// CK7:       [[EXECUTE:%.+]] = call i32 @__kmpc_master(%struct.ident_t* @0, i32 %1)
// CK7:       call void @__kmpc_end_master(%struct.ident_t* [[DEF_LOC_1]], i32 %1)

#endif
#ifdef CK8
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK8 -fopenmp -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

typedef __INTPTR_TYPE__ intptr_t;

// CK8-DAG: [[IDENT_T_TY:%.+]] = type { i32, i32, i32, i32, i8* }
// CK8-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK8-DAG: [[DEF_LOC_2:@.+]] = private unnamed_addr global [[IDENT_T_TY]] { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void foo();

struct S {
  intptr_t a, b, c;
  S(intptr_t a) : a(a) {}
  operator char() { return a; }
  ~S() {}
};

template <typename T>
T tmain() {
#pragma omp parallel master proc_bind(master)
  foo();
  return T();
}

int main() {
#pragma omp parallel master proc_bind(spread)
  foo();
#pragma omp parallel master proc_bind(close)
  foo();
  return tmain<int>();
}

// CK8-LABEL: @main
// CK8:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEF_LOC_2]])
// CK8:       call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 4)
// CK8:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CK8:       call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 3)
// CK8:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(

// CK8-LABEL: @{{.+}}tmain
// CK8:       [[GTID:%.+]] = call {{.*}}i32 @__kmpc_global_thread_num([[IDENT_T_TY]]* [[DEF_LOC_2]])
// CK8:       call {{.*}}void @__kmpc_push_proc_bind([[IDENT_T_TY]]* [[DEF_LOC_2]], i32 [[GTID]], i32 2)
// CK8:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(
// CK8:       ret i32 0
// CK8-NEXT:  }

// CK8:       call i32 @__kmpc_master(%struct.ident_t* [[DEF_LOC_2]], i32 [[ONE:%.+]])
// CK8:       call void @__kmpc_end_master(%struct.ident_t* [[DEF_LOC_2]], i32 [[ONE]])

#endif
#ifdef CK9
// CK9-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CK9-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CK9-DAG: [[DEF_LOC:@.+]] = private unnamed_addr global %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

void parallel_master_allocate() {
  int a;
#pragma omp parallel master firstprivate(a) allocate(a)
  a++;
}

// CK9-LABEL: define void @{{.+}}parallel_master_allocate{{.+}}
// CK9:       [[A_VAL:%.+]] = alloca i32
// CK9:       [[A_CASTED:%.+]] = alloca i64
// CK9:       [[ZERO:%.+]] = load i32, i32* [[A_VAL]]
// CK9:       [[CONV:%.+]] = bitcast i64* [[A_CASTED]] to i32*
// CK9:       store i32 [[ZERO]], i32* [[CONV]]
// CK9:       [[ONE:%.+]] = load i64, i64* [[A_CASTED]]
// CK9:       call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i64)* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i64 [[ONE]])

// CK9:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias [[GLOBAL_TID:%.+]], i32* noalias [[BOUND_TID:%.+]], i64 [[A_VAL]])
// CK9:       [[GLOBAL_TID_ADDR:%.+]] = alloca i32*
// CK9:       [[BOUND_TID_ADDR:%.+]] = alloca i32*
// CK9:       [[A_ADDR:%.+]] = alloca i64 
// CK9:       store i32* [[GLOBAL_TID]], i32** [[GLOBAL_TID_ADDR]]
// CK9:       store i32* [[BOUND_TID]], i32** [[BOUND_TID_ADDR]]
// CK9:       store i64 [[A_VAL]], i64* [[A_ADDR]]
// CK9:       [[CONV]] = bitcast i64* [[A_ADDR]] to i32*
// CK9-NOT:   __kmpc_global_thread_num
// CK9:       call i32 @__kmpc_master({{.+}})
// CK9:       [[FOUR:%.+]] = load i32, i32* [[CONV]]
// CK9:       [[INC:%.+]] = add nsw i32 [[FOUR]]
// CK9:       store i32 [[INC]], i32* [[CONV]]
// CK9-NOT:   __kmpc_global_thread_num
// CK9:       call void @__kmpc_end_master({{.+}})
#endif
#endif
