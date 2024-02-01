// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
// CK1: [[ST:%.+]] = type { i32, ptr }
// CK1: %struct.kmp_depend_info = type { i[[sz:64|32]],
// CK1-SAME: i[[sz]], i8 }
#ifndef HEADER
#define HEADER

template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] zeroinitializer

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 1]

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] zeroinitializer

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710673]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // CK1: alloca [1 x %struct.kmp_depend_info],
  // CK1: alloca [3 x %struct.kmp_depend_info],
  // CK1: alloca [4 x %struct.kmp_depend_info],
  // CK1: alloca [5 x %struct.kmp_depend_info],

  // Region 00
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1: store ptr @gc, ptr [[BP0]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1: store ptr @gc, ptr [[P0]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK1: [[CAP_DEVICE:%.+]] = getelementptr inbounds %struct.anon, ptr [[CAPTURES:%.+]], i32 0, i32 0
  // CK1: [[DEVICE:%.+]] = load i32, ptr %{{.+}}
  // CK1: store i32 [[DEVICE]], ptr [[CAP_DEVICE]],
  // CK1: [[DEV1:%.+]] = load i32, ptr %{{.+}}
  // CK1: [[DEV2:%.+]] = sext i32 [[DEV1]] to i64
  // CK1: [[RES:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{.+}}, i32 {{.+}}, i32 1, i[[sz]] {{64|36}}, i[[sz]] 4, ptr [[TASK_ENTRY0:@.+]], i64 [[DEV2]])
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates, ptr [[RES]], i32 0, i32 0
  // CK1: [[SHAREDS:%.+]] = getelementptr inbounds %struct.kmp_task_t, ptr [[TASK_T]], i32 0, i32 0
  // CK1: [[SHAREDS_REF:%.+]] = load ptr, ptr [[SHAREDS]],
  // CK1: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align 4 [[SHAREDS_REF]], ptr align 4 [[CAPTURES]], i[[sz]] 4, i1 false)
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates, ptr [[RES]], i32 0, i32 1
  // CK1-64: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 0
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 1
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 2
  // CK1-64: call void @llvm.memcpy.p0.p0.i64(ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[SIZE00]], i64 {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 0
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[SIZE00]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 1
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, ptr [[PRIVS]], i32 0, i32 2
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP:%.+]], i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: = call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]], i32 1, ptr [[MAIN_DEP]], i32 0, ptr null)

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target enter data if(1+3-5) device(arg) map(alloc:gc) nowait depend(in: arg)
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target enter data map(to:la) if(1+3-4) depend(in: la) depend(out: arg)
  {++arg;}

  // Region 02
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1: store ptr [[ARG:%.+]], ptr [[BP0]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1: store ptr [[ARG]], ptr [[P0]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK1: [[IF_DEVICE:%.+]] = getelementptr inbounds %struct.anon{{.+}}, ptr [[CAPTURES:%.+]], i32 0, i32 0
  // CK1: [[IF:%.+]] = load i8, ptr %{{.+}}
  // CK1: [[IF_BOOL:%.+]] = trunc i8 [[IF]] to i1
  // CK1: [[IF:%.+]] = zext i1 [[IF_BOOL]] to i8
  // CK1: store i8 [[IF]], ptr [[IF_DEVICE]],
  // CK1: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr {{.+}}, i32 {{.+}}, i32 1, i[[sz]] {{64|36}}, i[[sz]] 1, ptr [[TASK_ENTRY2:@.+]])
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 0
  // CK1: [[SHAREDS:%.+]] = getelementptr inbounds %struct.kmp_task_t, ptr [[TASK_T]], i32 0, i32 0
  // CK1: [[SHAREDS_REF:%.+]] = load ptr, ptr [[SHAREDS]],
  // CK1: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align 1 [[SHAREDS_REF]], ptr align 1 [[CAPTURES]], i[[sz]] 1, i1 false)
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 1
  // CK1-64: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[SIZE02]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[SIZE02]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP:%.+]], i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint (ptr @gc to i[[sz]]), ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %{{.+}}, i32 3, ptr [[MAIN_DEP]], i32 0, ptr null, i32 0)
  // CK1: call void @__kmpc_omp_task_begin_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  // CK1: = call i32 [[TASK_ENTRY2]](i32 %{{.+}}, ptr [[RES]])
  // CK1: call void @__kmpc_omp_task_complete_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])

  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target enter data map(to:arg) if(arg) device(4) depend(inout: arg, la, gc)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 03
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1: store ptr [[VLA:%.+]], ptr [[BP0]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1: store ptr [[VLA]], ptr [[P0]],
  // CK1: [[S0:%.+]] = getelementptr inbounds [1 x i64], ptr [[S:%.+]], i32 0, i32 0
  // CK1: store i64 {{.+}}, ptr [[S0]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK1: [[GEPS0:%.+]] = getelementptr inbounds [1 x i64], ptr [[S]], i32 0, i32 0
  // CK1: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr {{.+}}, i32 {{.+}}, i32 1, i[[sz]] {{64|36}}, i[[sz]] 1, ptr [[TASK_ENTRY3:@.+]])
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 0
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 1
  // CK1-64: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-64: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[GEPS0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[GEPS0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1-32: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{8|4}}, i1 false)
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP:%.+]], i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 3
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint (ptr @gc to i[[sz]]), ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 3, ptr [[DEP_ATTRS]]
  // CK1: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %{{.+}}, i32 4, ptr [[MAIN_DEP]], i32 0, ptr null, i32 0)
  // CK1: call void @__kmpc_omp_task_begin_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  // CK1: = call i32 [[TASK_ENTRY3]](i32 %{{.+}}, ptr [[RES]])
  // CK1: call void @__kmpc_omp_task_complete_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  #pragma omp target enter data map(alloc:lb) depend(out: lb, arg, la, gc)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1: [[DIV:%.+]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr (ptr, ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1) to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK1: [[BP0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1: store ptr @gb, ptr [[BP0]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1: store ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1), ptr [[P0]],
  // CK1: [[PS0:%.+]] = getelementptr inbounds [2 x i64], ptr [[PS:%.+]], i32 0, i32 0
  // CK1: store i64 [[DIV]], ptr [[PS0]],
  // CK1: [[BP1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 1
  // CK1: store ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1), ptr [[BP1]],
  // CK1: [[P1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 1
  // CK1: store ptr %{{.+}}, ptr [[P1]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 0
  // CK1: [[GEPS0:%.+]] = getelementptr inbounds [2 x i64], ptr [[PS]], i32 0, i32 0
  // CK1: [[RES:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr {{.+}}, i32 {{.+}}, i32 1, i[[sz]] {{88|52}}, i[[sz]] 1, ptr [[TASK_ENTRY4:@.+]])
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 0
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, ptr [[RES]], i32 0, i32 1
  // CK1-64: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{16|8}}, i1 false)
  // CK1-64: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{16|8}}, i1 false)
  // CK1-64: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-64: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[GEPS0]], i[[sz]] {{16|8}}, i1 false)
  // CK1-32: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_SIZES]], ptr align {{8|4}} [[GEPS0]], i[[sz]] {{16|8}}, i1 false)
  // CK1-32: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 1
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_BASEPTRS]], ptr align {{8|4}} [[GEPBP0]], i[[sz]] {{16|8}}, i1 false)
  // CK1-32: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, ptr [[PRIVS]], i32 0, i32 2
  // CK1-32: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{8|4}} [[PRIVS_PTRS]], ptr align {{8|4}} [[GEPP0]], i[[sz]] {{16|8}}, i1 false)
  // CK1: %{{.+}} = sub nuw
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP:%.+]], i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 3
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint (ptr @gc to i[[sz]]), ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: [[BC_ADR:%.+]] = ptrtoint ptr %{{.+}} to i[[sz]]
  // CK1: [[DEP:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[MAIN_DEP]], i[[sz]] 4
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] [[BC_ADR]], ptr [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, ptr [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, ptr [[DEP]], i32 0, i32 2
  // CK1: store i8 1, ptr [[DEP_ATTRS]]
  // CK1: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %{{.+}}, i32 5, ptr [[MAIN_DEP]], i32 0, ptr null, i32 0)
  // CK1: call void @__kmpc_omp_task_begin_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  // CK1: = call i32 [[TASK_ENTRY4]](i32 %{{.+}}, ptr [[RES]])
  // CK1: call void @__kmpc_omp_task_complete_if0(ptr @{{.+}}, i32 %{{.+}}, ptr [[RES]])
  #pragma omp target enter data map(to:gb.b[:3]) depend(in: gb.b[:3], la, lb, gc, arg)
  {++arg;}
}

// CK1: define internal{{.*}} i32 [[TASK_ENTRY0]](i32{{.*}}, ptr noalias noundef %1)
// CK1-DAG: call void @__tgt_target_data_begin_nowait_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 1, ptr [[BP:%.+]], ptr [[P:%.+]], ptr [[S:%.+]], ptr [[MTYPE00]]{{.+}}, ptr null)
// CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK1-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK1-DAG: [[BP]] = load ptr, ptr [[BP_PRIV:%[^,]+]],
// CK1-DAG: [[P]] = load ptr, ptr [[P_PRIV:%[^,]+]],
// CK1-DAG: [[S]] = load ptr, ptr [[S_PRIV:%[^,]+]],
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[BP_PRIV]], ptr [[P_PRIV]], ptr [[S_PRIV]])
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY2]](i32{{.*}}, ptr noalias noundef %1)
// CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 4, i32 1, ptr [[BP:%.+]], ptr [[P:%.+]], ptr [[S:%.+]], ptr [[MTYPE02]]{{.+}}, ptr null)
// CK1-DAG: [[BP]] = load ptr, ptr [[BP_PRIV:%[^,]+]],
// CK1-DAG: [[P]] = load ptr, ptr [[P_PRIV:%[^,]+]],
// CK1-DAG: [[S]] = load ptr, ptr [[S_PRIV:%[^,]+]],
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[BP_PRIV]], ptr [[P_PRIV]], ptr [[S_PRIV]])
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY3]](i32{{.*}}, ptr noalias noundef %1)
// CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[BP:%.+]], ptr [[P:%.+]], ptr [[S:%.+]], ptr [[MTYPE03]]{{.+}}, ptr null)
// CK1-DAG: [[BP]] = load ptr, ptr [[BP_PRIV:%[^,]+]],
// CK1-DAG: [[P]] = load ptr, ptr [[P_PRIV:%[^,]+]],
// CK1-DAG: [[S]] = load ptr, ptr [[S_PRIV:%[^,]+]],
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[BP_PRIV]], ptr [[P_PRIV]], ptr [[S_PRIV]])
// CK1-NOT: __tgt_target_data_end
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY4]](i32{{.*}}, ptr noalias noundef %1)
// CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[BP:%.+]], ptr [[P:%.+]], ptr [[S:%.+]], ptr [[MTYPE04]]{{.+}}, ptr null)
// CK1-DAG: [[BP]] = load ptr, ptr [[BP_PRIV:%[^,]+]],
// CK1-DAG: [[P]] = load ptr, ptr [[P_PRIV:%[^,]+]],
// CK1-DAG: [[S]] = load ptr, ptr [[S_PRIV:%[^,]+]],
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[BP_PRIV]], ptr [[P_PRIV]], ptr [[S_PRIV]])
// CK1-NOT: __tgt_target_data_end
// CK1: ret i32 0
// CK1: }

#endif
