// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// expected-no-diagnostics
// CK1: [[ST:%.+]] = type { i32, double* }
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

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 32]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 32]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i[[sz]]] [i[[sz]] {{8|4}}, i[[sz]] 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 33, i64 17]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // CK1: alloca [1 x %struct.kmp_depend_info],
  // CK1: alloca [3 x %struct.kmp_depend_info],
  // CK1: alloca [4 x %struct.kmp_depend_info],
  // CK1: alloca [5 x %struct.kmp_depend_info],

  // Region 00
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1: [[BP0_BC:%.+]] = bitcast i8** [[BP0]] to [100 x double]**
  // CK1: store [100 x double]* @gc, [100 x double]** [[BP0_BC]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1: [[P0_BC:%.+]] = bitcast i8** [[P0]] to [100 x double]**
  // CK1: store [100 x double]* @gc, [100 x double]** [[P0_BC]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK1: [[CAP_DEVICE:%.+]] = getelementptr inbounds %struct.anon, %struct.anon* [[CAPTURES:%.+]], i32 0, i32 0
  // CK1: [[DEVICE:%.+]] = load i32, i32* %{{.+}}
  // CK1: store i32 [[DEVICE]], i32* [[CAP_DEVICE]],
  // CK1: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%ident_t* {{.+}}, i32 {{.+}}, i32 1, i[[sz]] [[sz]], i[[sz]] 4, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates*)* [[TASK_ENTRY0:@.+]] to i32 (i32, i8*)*))
  // CK1: [[BC:%.+]] = bitcast i8* [[RES]] to %struct.kmp_task_t_with_privates*
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates, %struct.kmp_task_t_with_privates* [[BC]], i32 0, i32 0
  // CK1: [[SHAREDS:%.+]] = getelementptr inbounds %struct.kmp_task_t, %struct.kmp_task_t* [[TASK_T]], i32 0, i32 0
  // CK1: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS]],
  // CK1: [[BC1:%.+]] = bitcast %struct.anon* [[CAPTURES]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[SHAREDS_REF]], i8* [[BC1]], i[[sz]] 4, i32 4, i1 false)
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates, %struct.kmp_task_t_with_privates* [[BC]], i32 0, i32 1
  // CK1: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, %struct..kmp_privates.t* [[PRIVS]], i32 0, i32 0
  // CK1: [[BC_PRIVS_BASEPTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_BASEPTRS]] to i8*
  // CK1: [[BC_BASEPTRS:%.+]] = bitcast i8** [[GEPBP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_BASEPTRS]], i8* [[BC_BASEPTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t, %struct..kmp_privates.t* [[PRIVS]], i32 0, i32 1
  // CK1: [[BC_PRIVS_PTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_PTRS]] to i8*
  // CK1: [[BC_PTRS:%.+]] = bitcast i8** [[GEPP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_PTRS]], i8* [[BC_PTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t, %struct..kmp_privates.t* [[PRIVS]], i32 0, i32 2
  // CK1: [[BC_PRIVS_SIZES:%.+]] = bitcast [1 x i[[sz]]]* [[PRIVS_SIZES]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_SIZES]], i8* bitcast ([1 x i[[sz]]]* [[SIZE00]] to i8*), i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[DEP:%.+]] = getelementptr inbounds [1 x %struct.kmp_depend_info], [1 x %struct.kmp_depend_info]* [[MAIN_DEP:%.+]], i[[sz]] 0, i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [1 x %struct.kmp_depend_info], [1 x %struct.kmp_depend_info]* [[MAIN_DEP]], i32 0, i32 0
  // CK1: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[DEP]] to i8*
  // CK1: = call i32 @__kmpc_omp_task_with_deps(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]], i32 1, i8* [[BC]], i32 0, i8* null)

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
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1: [[BP0_BC:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK1: store i32* [[ARG:%.+]], i32** [[BP0_BC]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1: [[P0_BC:%.+]] = bitcast i8** [[P0]] to i32**
  // CK1: store i32* [[ARG]], i32** [[P0_BC]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK1: [[IF_DEVICE:%.+]] = getelementptr inbounds %struct.anon{{.+}}, %struct.anon{{.+}}* [[CAPTURES:%.+]], i32 0, i32 0
  // CK1: [[IF:%.+]] = load i8, i8* %{{.+}}
  // CK1: [[IF_BOOL:%.+]] = trunc i8 [[IF]] to i1
  // CK1: [[IF:%.+]] = zext i1 [[IF_BOOL]] to i8
  // CK1: store i8 [[IF]], i8* [[IF_DEVICE]],
  // CK1: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%ident_t* {{.+}}, i32 {{.+}}, i32 1, i[[sz]] [[sz]], i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates{{.+}}*)* [[TASK_ENTRY2:@.+]] to i32 (i32, i8*)*))
  // CK1: [[RES_BC:%.+]] = bitcast i8* [[RES]] to %struct.kmp_task_t_with_privates{{.+}}*
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 0
  // CK1: [[SHAREDS:%.+]] = getelementptr inbounds %struct.kmp_task_t, %struct.kmp_task_t* [[TASK_T]], i32 0, i32 0
  // CK1: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS]],
  // CK1: [[BC1:%.+]] = bitcast %struct.anon{{.+}}* [[CAPTURES]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[SHAREDS_REF]], i8* [[BC1]], i[[sz]] 1, i32 1, i1 false)
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 1
  // CK1: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 0
  // CK1: [[BC_PRIVS_BASEPTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_BASEPTRS]] to i8*
  // CK1: [[BC_BASEPTRS:%.+]] = bitcast i8** [[GEPBP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_BASEPTRS]], i8* [[BC_BASEPTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 1
  // CK1: [[BC_PRIVS_PTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_PTRS]] to i8*
  // CK1: [[BC_PTRS:%.+]] = bitcast i8** [[GEPP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_PTRS]], i8* [[BC_PTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 2
  // CK1: [[BC_PRIVS_SIZES:%.+]] = bitcast [1 x i[[sz]]]* [[PRIVS_SIZES]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_SIZES]], i8* bitcast ([1 x i[[sz]]]* [[SIZE02]] to i8*), i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[DEP:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[MAIN_DEP:%.+]], i[[sz]] 0, i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint ([100 x double]* @gc to i[[sz]]), i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [3 x %struct.kmp_depend_info], [3 x %struct.kmp_depend_info]* [[MAIN_DEP]], i32 0, i32 0
  // CK1: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[DEP]] to i8*
  // CK1: call void @__kmpc_omp_wait_deps(%ident_t* @{{.+}}, i32 %{{.+}}, i32 3, i8* [[BC]], i32 0, i8* null)
  // CK1: call void @__kmpc_omp_task_begin_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  // CK1: = call i32 [[TASK_ENTRY2]](i32 %{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]])
  // CK1: call void @__kmpc_omp_task_complete_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])

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
  // CK1: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1: [[BP0_BC:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1: store float* [[VLA:%.+]], float** [[BP0_BC]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1: [[P0_BC:%.+]] = bitcast i8** [[P0]] to float**
  // CK1: store float* [[VLA]], float** [[P0_BC]],
  // CK1: [[S0:%.+]] = getelementptr inbounds [1 x i[[sz]]], [1 x i[[sz]]]* [[S:%.+]], i32 0, i32 0
  // CK1: store i[[sz]] {{.+}}, i[[sz]]* [[S0]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK1: [[GEPS0:%.+]] = getelementptr inbounds [1 x i[[sz]]], [1 x i[[sz]]]* [[S]], i32 0, i32 0
  // CK1: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%ident_t* {{.+}}, i32 {{.+}}, i32 1, i[[sz]] [[sz]], i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates{{.+}}*)* [[TASK_ENTRY3:@.+]] to i32 (i32, i8*)*))
  // CK1: [[RES_BC:%.+]] = bitcast i8* [[RES]] to %struct.kmp_task_t_with_privates{{.+}}*
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 0
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 1
  // CK1: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 0
  // CK1: [[BC_PRIVS_BASEPTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_BASEPTRS]] to i8*
  // CK1: [[BC_BASEPTRS:%.+]] = bitcast i8** [[GEPBP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_BASEPTRS]], i8* [[BC_BASEPTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 1
  // CK1: [[BC_PRIVS_PTRS:%.+]] = bitcast [1 x i8*]* [[PRIVS_PTRS]] to i8*
  // CK1: [[BC_PTRS:%.+]] = bitcast i8** [[GEPP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_PTRS]], i8* [[BC_PTRS]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 2
  // CK1: [[BC_PRIVS_SIZES:%.+]] = bitcast [1 x i[[sz]]]* [[PRIVS_SIZES]] to i8*
  // CK1: [[BC_SIZES:%.+]] = bitcast i[[sz]]* [[GEPS0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_SIZES]], i8* [[BC_SIZES]], i[[sz]] {{8|4}}, i32 {{8|4}}, i1 false)
  // CK1: [[DEP:%.+]] = getelementptr inbounds [4 x %struct.kmp_depend_info], [4 x %struct.kmp_depend_info]* [[MAIN_DEP:%.+]], i[[sz]] 0, i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint float* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [4 x %struct.kmp_depend_info], [4 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [4 x %struct.kmp_depend_info], [4 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [4 x %struct.kmp_depend_info], [4 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 3
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint ([100 x double]* @gc to i[[sz]]), i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 3, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [4 x %struct.kmp_depend_info], [4 x %struct.kmp_depend_info]* [[MAIN_DEP]], i32 0, i32 0
  // CK1: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[DEP]] to i8*
  // CK1: call void @__kmpc_omp_wait_deps(%ident_t* @{{.+}}, i32 %{{.+}}, i32 4, i8* [[BC]], i32 0, i8* null)
  // CK1: call void @__kmpc_omp_task_begin_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  // CK1: = call i32 [[TASK_ENTRY3]](i32 %{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]])
  // CK1: call void @__kmpc_omp_task_complete_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  #pragma omp target enter data map(alloc:lb) depend(out: lb, arg, la, gc)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1: [[BP0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1: [[BP0_BC:%.+]] = bitcast i8** [[BP0]] to %struct.ST**
  // CK1: store %struct.ST* @gb, %struct.ST** [[BP0_BC]],
  // CK1: [[P0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1: [[P0_BC:%.+]] = bitcast i8** [[P0]] to double***
  // CK1: store double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), double*** [[P0_BC]],
  // CK1: [[BP1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 1
  // CK1: [[BP1_BC:%.+]] = bitcast i8** [[BP1]] to double***
  // CK1: store double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), double*** [[BP1_BC]],
  // CK1: [[P1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 1
  // CK1: [[P1_BC:%.+]] = bitcast i8** [[P1]] to double**
  // CK1: store double* %{{.+}}, double** [[P1_BC]],
  // CK1: [[GEPBP0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 0
  // CK1: [[GEPP0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 0
  // CK1: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%ident_t* {{.+}}, i32 {{.+}}, i32 1, i[[sz]] {{88|44}}, i[[sz]] 1, i32 (i32, i8*)* bitcast (i32 (i32, %struct.kmp_task_t_with_privates{{.+}}*)* [[TASK_ENTRY4:@.+]] to i32 (i32, i8*)*))
  // CK1: [[RES_BC:%.+]] = bitcast i8* [[RES]] to %struct.kmp_task_t_with_privates{{.+}}*
  // CK1: [[TASK_T:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 0
  // CK1: [[PRIVS:%.+]] = getelementptr inbounds %struct.kmp_task_t_with_privates{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]], i32 0, i32 1
  // CK1: [[PRIVS_BASEPTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 0
  // CK1: [[BC_PRIVS_BASEPTRS:%.+]] = bitcast [2 x i8*]* [[PRIVS_BASEPTRS]] to i8*
  // CK1: [[BC_BASEPTRS:%.+]] = bitcast i8** [[GEPBP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_BASEPTRS]], i8* [[BC_BASEPTRS]], i[[sz]] {{16|8}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_PTRS:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 1
  // CK1: [[BC_PRIVS_PTRS:%.+]] = bitcast [2 x i8*]* [[PRIVS_PTRS]] to i8*
  // CK1: [[BC_PTRS:%.+]] = bitcast i8** [[GEPP0]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_PTRS]], i8* [[BC_PTRS]], i[[sz]] {{16|8}}, i32 {{8|4}}, i1 false)
  // CK1: [[PRIVS_SIZES:%.+]] = getelementptr inbounds %struct..kmp_privates.t{{.+}}, %struct..kmp_privates.t{{.+}}* [[PRIVS]], i32 0, i32 2
  // CK1: [[BC_PRIVS_SIZES:%.+]] = bitcast [2 x i[[sz]]]* [[PRIVS_SIZES]] to i8*
  // CK1: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* [[BC_PRIVS_SIZES]], i8* bitcast ([2 x i[[sz]]]* [[SIZE04]] to i8*), i[[sz]] {{16|8}}, i32 {{8|4}}, i1 false)
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP:%.+]], i[[sz]] 0, i[[sz]] 0
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint double* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 1
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 2
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint float* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] %{{.+}}, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 3
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: store i[[sz]] ptrtoint ([100 x double]* @gc to i[[sz]]), i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 800, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP]], i[[sz]] 0, i[[sz]] 4
  // CK1: [[DEP_ADR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 0
  // CK1: [[BC_ADR:%.+]] = ptrtoint i32* %{{.+}} to i[[sz]]
  // CK1: store i[[sz]] [[BC_ADR]], i[[sz]]* [[DEP_ADR]],
  // CK1: [[DEP_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 1
  // CK1: store i[[sz]] 4, i[[sz]]* [[DEP_SIZE]],
  // CK1: [[DEP_ATTRS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEP]], i32 0, i32 2
  // CK1: store i8 1, i8* [[DEP_ATTRS]]
  // CK1: [[DEP:%.+]] = getelementptr inbounds [5 x %struct.kmp_depend_info], [5 x %struct.kmp_depend_info]* [[MAIN_DEP]], i32 0, i32 0
  // CK1: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[DEP]] to i8*
  // CK1: call void @__kmpc_omp_wait_deps(%ident_t* @{{.+}}, i32 %{{.+}}, i32 5, i8* [[BC]], i32 0, i8* null)
  // CK1: call void @__kmpc_omp_task_begin_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  // CK1: = call i32 [[TASK_ENTRY4]](i32 %{{.+}}, %struct.kmp_task_t_with_privates{{.+}}* [[RES_BC]])
  // CK1: call void @__kmpc_omp_task_complete_if0(%ident_t* @{{.+}}, i32 %{{.+}}, i8* [[RES]])
  #pragma omp target enter data map(to:gb.b[:3]) depend(in: gb.b[:3], la, lb, gc, arg)
  {++arg;}
}

// CK1: define internal{{.*}} i32 [[TASK_ENTRY0]](i32{{.*}}, %struct.kmp_task_t_with_privates* noalias)
// CK1-DAG: call void @__tgt_target_data_begin_nowait(i64 [[DEV:%[^,]+]], i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
// CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK1-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
// CK1-DAG: [[BP]] = load [1 x i8*]*, [1 x i8*]** [[BP_PRIV:%.+]],
// CK1-DAG: [[P]] = load [1 x i8*]*, [1 x i8*]** [[P_PRIV:%.+]],
// CK1-DAG: [[S]] = load [1 x i[[sz]]]*, [1 x i[[sz]]]** [[S_PRIV:%.+]],
// CK1-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[BP_PRIV]], [1 x i8*]** [[P_PRIV]], [1 x i[[sz]]]** [[S_PRIV]])
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY2]](i32{{.*}}, %struct.kmp_task_t_with_privates{{.+}}* noalias)
// CK1-DAG: call void @__tgt_target_data_begin(i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
// CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
// CK1-DAG: [[BP]] = load [1 x i8*]*, [1 x i8*]** [[BP_PRIV:%.+]],
// CK1-DAG: [[P]] = load [1 x i8*]*, [1 x i8*]** [[P_PRIV:%.+]],
// CK1-DAG: [[S]] = load [1 x i[[sz]]]*, [1 x i[[sz]]]** [[S_PRIV:%.+]],
// CK1-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[BP_PRIV]], [1 x i8*]** [[P_PRIV]], [1 x i[[sz]]]** [[S_PRIV]])
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY3]](i32{{.*}}, %struct.kmp_task_t_with_privates{{.+}}* noalias)
// CK1-DAG: call void @__tgt_target_data_begin(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
// CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK1-DAG: [[BP]] = load [1 x i8*]*, [1 x i8*]** [[BP_PRIV:%.+]],
// CK1-DAG: [[P]] = load [1 x i8*]*, [1 x i8*]** [[P_PRIV:%.+]],
// CK1-DAG: [[S]] = load [1 x i[[sz]]]*, [1 x i[[sz]]]** [[S_PRIV:%.+]],
// CK1-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[BP_PRIV]], [1 x i8*]** [[P_PRIV]], [1 x i[[sz]]]** [[S_PRIV]])
// CK1-NOT: __tgt_target_data_end
// CK1: ret i32 0
// CK1: }

// CK1: define internal{{.*}} i32 [[TASK_ENTRY4]](i32{{.*}}, %struct.kmp_task_t_with_privates{{.+}}* noalias)
// CK1-DAG: call void @__tgt_target_data_begin(i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}})
// CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK1-DAG: [[BP]] = load [2 x i8*]*, [2 x i8*]** [[BP_PRIV:%.+]],
// CK1-DAG: [[P]] = load [2 x i8*]*, [2 x i8*]** [[P_PRIV:%.+]],
// CK1-DAG: [[S]] = load [2 x i[[sz]]]*, [2 x i[[sz]]]** [[S_PRIV:%.+]],
// CK1-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [2 x i8*]** [[BP_PRIV]], [2 x i8*]** [[P_PRIV]], [2 x i[[sz]]]** [[S_PRIV]])
// CK1-NOT: __tgt_target_data_end
// CK1: ret i32 0
// CK1: }

#endif
