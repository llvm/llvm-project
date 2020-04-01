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
  int *argv;
#pragma omp depobj(a) depend(in:argv, ([3][*(int*)argv][4])argv)
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
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[MAIN_A]],
// CHECK: [[B:%.+]] = load i8*, i8** [[B_ADDR]],
// CHECK: [[B_BASE:%.+]] = bitcast i8* [[B]] to %struct.kmp_depend_info*
// CHECK: [[B_REF:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i{{.+}} -1
// CHECK: [[B:%.+]] = bitcast %struct.kmp_depend_info* [[B_REF]] to i8*
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[B]], i8* null)
// CHECK: [[B:%.+]] = load i8*, i8** [[B_ADDR]],
// CHECK: [[B_BASE:%.+]] = bitcast i8* [[B]] to %struct.kmp_depend_info*
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, i64* [[NUMDEPS_ADDR]],
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[B_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi %struct.kmp_depend_info* [ [[B_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 4, i8* [[FLAG_BASE]],
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq %struct.kmp_depend_info* [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

// CHECK-LABEL: tmain
// CHECK: [[ARGC_ADDR:%.+]] = alloca i8*,
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
// CHECK: store i64 8, i64* [[SZ_ADDR]],
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, i8* [[FLAGS_ADDR]],
// CHECK: [[SHAPE_ADDR:%.+]] = load i32*, i32** [[ARGV_ADDR:%.+]],
// CHECK: [[SZ1:%.+]] = mul nuw i64 12, %{{.+}}
// CHECK: [[SZ:%.+]] = mul nuw i64 [[SZ1]], 4
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: [[ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SHAPE:%.+]] = ptrtoint i32* [[SHAPE_ADDR]] to i64
// CHECK: store i64 [[SHAPE]], i64* [[ADDR]],
// CHECK: [[SZ_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i64 [[SZ]], i64* [[SZ_ADDR]],
// CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[BASE_ADDR]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 1, i8* [[FLAGS_ADDR]],
// CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[DEP_ADDR]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[DEP:%.+]] = bitcast %struct.kmp_depend_info* [[BASE_ADDR]] to i8*
// CHECK: store i8* [[DEP]], i8** [[TMAIN_A]],
// CHECK: [[ARGC:%.+]] = load i8*, i8** [[ARGC_ADDR]],
// CHECK: [[ARGC_BASE:%.+]] = bitcast i8* [[ARGC]] to %struct.kmp_depend_info*
// CHECK: [[ARGC_REF:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i{{.+}} -1
// CHECK: [[ARGC:%.+]] = bitcast %struct.kmp_depend_info* [[ARGC_REF]] to i8*
// CHECK: call void @__kmpc_free(i32 [[GTID]], i8* [[ARGC]], i8* null)
// CHECK: [[ARGC:%.+]] = load i8*, i8** [[ARGC_ADDR]],
// CHECK: [[ARGC_BASE:%.+]] = bitcast i8* [[ARGC]] to %struct.kmp_depend_info*
// CHECK: [[NUMDEPS_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i64 -1
// CHECK: [[NUMDEPS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[NUMDEPS_BASE]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[NUMDEPS:%.+]] = load i64, i64* [[NUMDEPS_ADDR]],
// CHECK: [[END:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[ARGC_BASE]], i64 [[NUMDEPS]]
// CHECK: br label %[[BODY:.+]]
// CHECK: [[BODY]]:
// CHECK: [[EL:%.+]] = phi %struct.kmp_depend_info* [ [[ARGC_BASE]], %{{.+}} ], [ [[EL_NEXT:%.+]], %[[BODY]] ]
// CHECK: [[FLAG_BASE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i8 3, i8* [[FLAG_BASE]],
// CHECK: [[EL_NEXT]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[EL]], i{{.+}} 1
// CHECK: [[IS_DONE:%.+]] = icmp eq %struct.kmp_depend_info* [[EL_NEXT]], [[END]]
// CHECK: br i1 [[IS_DONE]], label %[[DONE:.+]], label %[[BODY]]
// CHECK: [[DONE]]:

#endif
