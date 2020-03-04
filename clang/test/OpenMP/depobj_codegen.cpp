// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=50 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10 -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -triple x86_64-apple-darwin10 -fopenmp-version=50 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=50 -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10 -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -triple x86_64-apple-darwin10 -fopenmp-version=50 -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-DAG: [[MAIN_A:@.+]] = internal global i8* null,
// CHECK-DAG: [[TMAIN_A:@.+]] = linkonce_odr global i8* null,

typedef void *omp_depend_t;

void foo() {}

template <class T>
T tmain(T argc) {
  static T a;
#pragma omp depobj(a) depend(in:argc)
#pragma omp depobj(argc) destroy
#pragma omp depobj(argc) update(inout)
  return argc;
}

int main(int argc, char **argv) {
  static omp_depend_t a;
  omp_depend_t b;
#pragma omp depobj(a) depend(out:argc, argv)
#pragma omp depobj(b) destroy
#pragma omp depobj(b) update(mutexinoutset)
  (void)tmain(a), tmain(b);
  return 0;
}

// CHECK-LABEL: @main
// CHECK: [[B_ADDR:%.+]] = alloca i8*,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID]], i64 72, i8* null)
// CHECK: [[DEP_ADDR:%.+]] = bitcast i8* [[DEP_ADDR_VOID]] to [3 x %struct.kmp_depend_info]*
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 2, i64* [[SZ_BASE]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]],
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 4, i64* [[SZ_ADDR]],
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAGS_ADDR]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]],
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, i64* [[SZ_ADDR]],
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAGS_ADDR]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[MAIN_A]],
// CHECK: [[B:%.+]] = load i8*, i8** [[B_ADDR]],
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[B]], i8* null)

// CHECK-LABEL: tmain
// CHECK: [[ARGC_ADDR:%.+]] = alloca i8*,
// CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
// CHECK: [[DEP_ADDR_VOID:%.+]] = call i8* @__kmpc_alloc(i32 [[GTID]], i64 48, i8* null)
// CHECK: [[DEP_ADDR:%.+]] = bitcast i8* [[DEP_ADDR_VOID]] to [2 x %struct.kmp_depend_info]*
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [2 x %struct.kmp_depend_info], [2 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SZ_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 1, i64* [[SZ_BASE]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [2 x %struct.kmp_depend_info], [2 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: store i64 %{{.+}}, i64* [[ADDR]],
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 8, i64* [[SZ_ADDR]],
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, i8* [[FLAGS_ADDR]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [2 x %struct.kmp_depend_info], [2 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[TMAIN_A]],
// CHECK: [[ARGC:%.+]] = load i8*, i8** [[ARGC_ADDR]],
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[ARGC]], i8* null)

#endif
