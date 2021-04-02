// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - -DUNTIEDRT | FileCheck %s --check-prefix CHECK --check-prefix UNTIEDRT
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s -DUNTIEDRT
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix UNTIEDRT
//
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-enable-irbuilder -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-enable-irbuilder -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

enum omp_allocator_handle_t {
  omp_null_allocator = 0,
  omp_default_mem_alloc = 1,
  omp_large_cap_mem_alloc = 2,
  omp_const_mem_alloc = 3,
  omp_high_bw_mem_alloc = 4,
  omp_low_lat_mem_alloc = 5,
  omp_cgroup_mem_alloc = 6,
  omp_pteam_mem_alloc = 7,
  omp_thread_mem_alloc = 8,
  KMP_ALLOCATOR_MAX_HANDLE = __UINTPTR_MAX__
};

// CHECK-DAG: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STRUCT_SHAREDS:%.+]] = type { i8*, [2 x [[STRUCT_S:%.+]]]* }
// CHECK-DAG: [[STRUCT_SHAREDS1:%.+]] = type { [2 x [[STRUCT_S:%.+]]]* }
// CHECK-DAG: [[KMP_TASK_T:%.+]] = type { i8*, i32 (i32, i8*)*, i32, %union{{.+}}, %union{{.+}} }
// CHECK-DAG: [[KMP_DEPEND_INFO:%.+]] = type { i64, i64, i8 }
struct S {
  int a;
  S() : a(0) {}
  S(const S &s) : a(s.a) {}
  ~S() {}
};
int a;
// CHECK-LABEL: @main
int main() {
// CHECK: [[B:%.+]] = alloca i8
// CHECK: [[S:%.+]] = alloca [2 x [[STRUCT_S]]]
  char b;
  S s[2];
  int arr[10][a];
// CHECK: [[B_REF:%.+]] = getelementptr inbounds [[STRUCT_SHAREDS]], [[STRUCT_SHAREDS]]* [[CAPTURES:%.+]], i32 0, i32 0
// CHECK: store i8* [[B]], i8** [[B_REF]]
// CHECK: [[S_REF:%.+]] = getelementptr inbounds [[STRUCT_SHAREDS]], [[STRUCT_SHAREDS]]* [[CAPTURES]], i32 0, i32 1
// CHECK: store [2 x [[STRUCT_S]]]* [[S]], [2 x [[STRUCT_S]]]** [[S_REF]]
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 33, i64 40, i64 16, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY1:@.+]] to i32 (i32, i8*)*))
// CHECK: [[SHAREDS_REF_PTR:%.+]] = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASK_PTR:%.+]], i32 0, i32 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_PTR]]
// CHECK: [[BITCAST:%.+]] = bitcast [[STRUCT_SHAREDS]]* [[CAPTURES]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[SHAREDS_REF]], i8* align 8 [[BITCAST]], i64 16, i1 false)
// CHECK: [[PRIORITY_REF_PTR:%.+]] = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASK_PTR]], i32 0, i32 4
// CHECK: [[PRIORITY:%.+]] = bitcast %union{{.+}}* [[PRIORITY_REF_PTR]] to i32*
// CHECK: store i32 {{.+}}, i32* [[PRIORITY]]
// CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
#pragma omp task shared(a, b, s) priority(b)
  {
    a = 15;
    b = a;
    s[0].a = 10;
  }
// CHECK: [[S_REF:%.+]] = getelementptr inbounds [[STRUCT_SHAREDS1]], [[STRUCT_SHAREDS1]]* [[CAPTURES:%.+]], i32 0, i32 0
// CHECK: store [2 x [[STRUCT_S]]]* [[S]], [2 x [[STRUCT_S]]]** [[S_REF]]
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 {{%.*}}, i32 1, i64 40, i64 8,
// CHECK: [[SHAREDS_REF_PTR:%.+]] = getelementptr inbounds [[KMP_TASK_T]], [[KMP_TASK_T]]* [[TASK_PTR:%.+]], i32 0, i32 0
// CHECK: [[SHAREDS_REF:%.+]] = load i8*, i8** [[SHAREDS_REF_PTR]]
// CHECK: [[BITCAST:%.+]] = bitcast [[STRUCT_SHAREDS1]]* [[CAPTURES]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[SHAREDS_REF]], i8* align 8 [[BITCAST]], i64 8, i1 false)
// CHECK: [[DEP_BASE:%.*]] = getelementptr inbounds [4 x [[KMP_DEPEND_INFO]]], [4 x [[KMP_DEPEND_INFO]]]* [[DEPENDENCIES:%.*]], i64 0, i64 0
// CHECK: [[DEP:%.+]] = getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP_BASE]], i64 0
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 0
// CHECK: store i64 ptrtoint (i32* @{{.+}} to i64), i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 1
// CHECK: store i64 4, i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 2
// CHECK: store i8 1, i8* [[T0]]
// CHECK: [[DEP:%.*]] = getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP_BASE]], i64 1
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 0
// CHECK: ptrtoint i8* [[B]] to i64
// CHECK: store i64 %{{[^,]+}}, i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 1
// CHECK: store i64 1, i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 2
// CHECK: store i8 1, i8* [[T0]]
// CHECK: [[DEP:%.*]] = getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP_BASE]], i64 2
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 0
// CHECK: ptrtoint [2 x [[STRUCT_S]]]* [[S]] to i64
// CHECK: store i64 %{{[^,]+}}, i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 1
// CHECK: store i64 8, i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 2
// CHECK: store i8 1, i8* [[T0]]
// CHECK: [[IDX1:%.+]] = mul nsw i64 0, [[A_VAL:%.+]]
// CHECK: [[START:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[IDX1:%.+]] = mul nsw i64 9, [[A_VAL]]
// CHECK: [[END:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[END1:%.+]] = getelementptr i32, i32* [[END]], i32 1
// CHECK: [[START_INT:%.+]] = ptrtoint i32* [[START]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i32* [[END1]] to i64
// CHECK: [[SIZEOF:%.+]] = sub nuw i64 [[END_INT]], [[START_INT]]
// CHECK: [[DEP:%.*]] = getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP_BASE]], i64 3
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* [[DEP]], i32 0, i32 0
// CHECK: [[T1:%.*]] = ptrtoint i32* [[START]] to i64
// CHECK: store i64 [[T1]], i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 [[SIZEOF]], i64* [[T0]]
// CHECK: [[T0:%.*]] = getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 1, i8* [[T0]]
// CHECK: bitcast [[KMP_DEPEND_INFO]]* [[DEP_BASE]] to i8*
// CHECK: call i32 @__kmpc_omp_task_with_deps([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]], i32 4, i8* %{{[^,]+}}, i32 0, i8* null)
#pragma omp task shared(a, s) depend(in : a, b, s, arr[:])
  {
    a = 15;
    s[1].a = 10;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 0, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY2:@.+]] to i32 (i32, i8*)*))
// CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
#pragma omp task untied
  {
#pragma omp critical
    a = 1;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 0, i64 40, i64 1,
// CHECK: getelementptr inbounds [2 x [[STRUCT_S]]], [2 x [[STRUCT_S]]]* [[S]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint [[STRUCT_S]]* %{{.+}} to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 4, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 3, i8*
// CHECK: [[B_VAL:%.+]] = load i8, i8* [[B]]
// CHECK: [[IDX2:%.+]] = sext i8 [[B_VAL]] to i64
// CHECK: [[IDX1:%.+]] = mul nsw i64 4, [[A_VAL]]
// CHECK: [[START:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[START1:%.+]] = getelementptr inbounds i32, i32* [[START]], i64 [[IDX2]]
// CHECK: [[B_VAL:%.+]] = load i8, i8* [[B]]
// CHECK: [[IDX2:%.+]] = sext i8 [[B_VAL]] to i64
// CHECK: [[IDX1:%.+]] = mul nsw i64 9, [[A_VAL]]
// CHECK: [[END:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[END1:%.+]] = getelementptr inbounds i32, i32* [[END]], i64 [[IDX2]]
// CHECK: [[END2:%.+]] = getelementptr i32, i32* [[END1]], i32 1
// CHECK: [[START_INT:%.+]] = ptrtoint i32* [[START1]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i32* [[END2]] to i64
// CHECK: [[SIZEOF:%.+]] = sub nuw i64 [[END_INT]], [[START_INT]]
// CHECK: getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i64 1
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint i32* [[START1]] to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 [[SIZEOF]], i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 3, i8*
// CHECK: bitcast [[KMP_DEPEND_INFO]]* %{{.+}} to i8*
// CHECK: call i32 @__kmpc_omp_task_with_deps([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]], i32 2, i8* %{{[^,]+}}, i32 0, i8* null)
#pragma omp task untied depend(out : s[0], arr[4:][b])
  {
    a = 1;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 0, i64 40, i64 1,
// CHECK: getelementptr inbounds [2 x [[STRUCT_S]]], [2 x [[STRUCT_S]]]* [[S]], i64 0, i64 0
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint [[STRUCT_S]]* %{{.+}} to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 4, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 4, i8*
// CHECK: [[B_VAL:%.+]] = load i8, i8* [[B]]
// CHECK: [[IDX2:%.+]] = sext i8 [[B_VAL]] to i64
// CHECK: [[IDX1:%.+]] = mul nsw i64 4, [[A_VAL]]
// CHECK: [[START:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[START1:%.+]] = getelementptr inbounds i32, i32* [[START]], i64 [[IDX2]]
// CHECK: [[B_VAL:%.+]] = load i8, i8* [[B]]
// CHECK: [[IDX2:%.+]] = sext i8 [[B_VAL]] to i64
// CHECK: [[IDX1:%.+]] = mul nsw i64 9, [[A_VAL]]
// CHECK: [[END:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[END1:%.+]] = getelementptr inbounds i32, i32* [[END]], i64 [[IDX2]]
// CHECK: [[END2:%.+]] = getelementptr i32, i32* [[END1]], i32 1
// CHECK: [[START_INT:%.+]] = ptrtoint i32* [[START1]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i32* [[END2]] to i64
// CHECK: [[SIZEOF:%.+]] = sub nuw i64 [[END_INT]], [[START_INT]]
// CHECK: getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i64 1
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint i32* [[START1]] to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 [[SIZEOF]], i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 4, i8*
// CHECK: bitcast [[KMP_DEPEND_INFO]]* %{{.+}} to i8*
// CHECK: call i32 @__kmpc_omp_task_with_deps([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]], i32 2, i8* %{{[^,]+}}, i32 0, i8* null)
#pragma omp task untied depend(mutexinoutset: s[0], arr[4:][b])
  {
    a = 1;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 3, i64 40, i64 1,
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: store i64 ptrtoint (i32* @{{.+}} to i64), i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 4, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 3, i8*
// CHECK: getelementptr inbounds [2 x [[STRUCT_S]]], [2 x [[STRUCT_S]]]* [[S]], i64 0, i64 1
// CHECK: getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i64 1
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint [[STRUCT_S]]* %{{.+}} to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 4, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 3, i8*
// CHECK: [[IDX1:%.+]] = mul nsw i64 0, [[A_VAL]]
// CHECK: [[START:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[START1:%.+]] = getelementptr inbounds i32, i32* [[START]], i64 3
// CHECK: [[NEW_A_VAL:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[NEW_A_VAL_I64:%.+]] = sext i32 [[NEW_A_VAL]] to i64
// CHECK: [[IDX2:%.+]] = sub nsw i64 [[NEW_A_VAL_I64]], 1
// CHECK: [[NEW_A_VAL:%.+]] = load i32, i32* @{{.+}},
// CHECK: [[NEW_A_VAL_I64:%.+]] = sext i32 [[NEW_A_VAL]] to i64
// CHECK: [[SUB:%.+]] = add nsw i64 -1, [[NEW_A_VAL_I64]]
// CHECK: [[IDX1:%.+]] = mul nsw i64 [[SUB]], [[A_VAL]]
// CHECK: [[END:%.+]] = getelementptr inbounds i32, i32* %{{.+}}, i64 [[IDX1]]
// CHECK: [[END1:%.+]] = getelementptr inbounds i32, i32* [[END]], i64 [[IDX2]]
// CHECK: [[END2:%.+]] = getelementptr i32, i32* [[END1]], i32 1
// CHECK: [[START_INT:%.+]] = ptrtoint i32* [[START1]] to i64
// CHECK: [[END_INT:%.+]] = ptrtoint i32* [[END2]] to i64
// CHECK: [[SIZEOF:%.+]] = sub nuw i64 [[END_INT]], [[START_INT]]
// CHECK: getelementptr [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i64 2
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 0
// CHECK: ptrtoint i32* [[START1]] to i64
// CHECK: store i64 %{{[^,]+}}, i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 1
// CHECK: store i64 [[SIZEOF]], i64*
// CHECK: getelementptr inbounds [[KMP_DEPEND_INFO]], [[KMP_DEPEND_INFO]]* %{{[^,]+}}, i32 0, i32 2
// CHECK: store i8 3, i8*
// CHECK: bitcast [[KMP_DEPEND_INFO]]* %{{.+}} to i8*
// CHECK: call i32 @__kmpc_omp_task_with_deps([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]], i32 3, i8* %{{[^,]+}}, i32 0, i8* null)
#pragma omp task final(true) depend(inout: a, s[1], arr[:a][3:])
  {
    a = 2;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 3, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY3:@.+]] to i32 (i32, i8*)*))
// CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
#pragma omp task final(true)
  {
    a = 2;
  }
  // CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 1, i64 40, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY4:@.+]] to i32 (i32, i8*)*))
  // CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
  const bool flag = false;
#pragma omp task final(flag)
  {
    a = 3;
  }
  // CHECK: [[B_VAL:%.+]] = load i8, i8* [[B]]
  // CHECK: [[CMP:%.+]] = icmp ne i8 [[B_VAL]], 0
  // CHECK: [[FINAL:%.+]] = select i1 [[CMP]], i32 2, i32 0
  // CHECK: [[FLAGS:%.+]] = or i32 [[FINAL]], 1
  // CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 [[FLAGS]], i64 40, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY5:@.+]] to i32 (i32, i8*)*))
  // CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
  int c __attribute__((aligned(128)));
#pragma omp task final(b) shared(c)
  {
    a = 4;
    c = 5;
  }
// CHECK: [[ORIG_TASK_PTR:%.+]] = call i8* @__kmpc_omp_task_alloc([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i32 0, i64 256, i64 1, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T]]{{.*}}*)* [[TASK_ENTRY6:@.+]] to i32 (i32, i8*)*))
// CHECK: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{.+}}, i32 {{%.*}}, i8* [[ORIG_TASK_PTR]])
#pragma omp task untied firstprivate(c) allocate(omp_pteam_mem_alloc:c)
  {
    S s1, s2;
#ifdef UNTIEDRT
#pragma omp allocate(s2) allocator(omp_pteam_mem_alloc)
#endif
    s2.a = 0;
#pragma omp task
    a = c = 4;
#pragma omp taskyield
    s1 = S();
    s2.a = 10;
#pragma omp taskwait
  }
  return a;
}
// CHECK: define internal i32 [[TASK_ENTRY1]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %1)
// CHECK: store i32 15, i32* [[A_PTR:@.+]],
// CHECK: [[A_VAL:%.+]] = load i32, i32* [[A_PTR]]
// CHECK: [[A_VAL_I8:%.+]] = trunc i32 [[A_VAL]] to i8
// CHECK: store i8 [[A_VAL_I8]], i8* %{{.+}}
// CHECK: store i32 10, i32* %{{.+}}

// CHECK: define internal i32 [[TASK_ENTRY2]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %1)
// CHECK: store i32 1, i32* [[A_PTR]]

// CHECK: define internal i32 [[TASK_ENTRY3]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %1)
// CHECK: store i32 2, i32* [[A_PTR]]

// CHECK: define internal i32 [[TASK_ENTRY4]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %1)
// CHECK: store i32 3, i32* [[A_PTR]]

// CHECK: define internal i32 [[TASK_ENTRY5]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %1)
// CHECK: store i32 4, i32* [[A_PTR]]
// CHECK: store i32 5, i32* [[C_PTR:%.+]], align 128

// CHECK: define internal i32
// CHECK: store i32 4, i32* [[A_PTR]]

// CHECK: define internal i32 [[TASK_ENTRY6]](i32 %0, [[KMP_TASK_T]]{{.*}}* noalias %{{.+}})
// UNTIEDRT: [[S1_ADDR_PTR:%.+]] = alloca %struct.S*,
// UNTIEDRT: [[S2_ADDR_PTR_REF:%.+]] = alloca %struct.S**,
// UNTIEDRT: [[FN:%.+]] = bitcast void (i8*, ...)* %{{.*}} to void (i8*,
// UNTIEDRT: call void [[FN]](i8* %{{.+}}, %struct.S** [[S1_ADDR_PTR]], %struct.S*** [[S2_ADDR_PTR_REF]])
// UNTIEDRT-DAG: [[S1_ADDR:%.+]] = load %struct.S*, %struct.S** [[S1_ADDR_PTR]],
// UNTIEDRT-DAG: [[S2_ADDR_PTR:%.+]] = load %struct.S**, %struct.S*** [[S2_ADDR_PTR_REF]],
// UNTIEDRT-DAG: [[S2_ADDR:%.+]] = load %struct.S*, %struct.S** [[S2_ADDR_PTR]],
// CHECK: switch i32 %{{.+}}, label %[[DONE:.+]] [

// CHECK: [[DONE]]:
// CHECK: br label %[[CLEANUP:[^,]+]]

// CHECK: load i32*, i32** %
// CHECK: store i32 1, i32* %
// CHECK: call i32 @__kmpc_omp_task(%
// UNTIEDRT: br label %[[EXIT:[^,]+]]

// UNTIEDRT: call void [[CONSTR:@.+]](%struct.S* {{[^,]*}} [[S1_ADDR]])
// UNTIEDRT: [[S2_VOID_PTR:%.+]] = call i8* @__kmpc_alloc(i32 %{{.+}}, i64 4, i8* inttoptr (i64 7 to i8*))
// UNTIEDRT: [[S2_PTR:%.+]] = bitcast i8* [[S2_VOID_PTR]] to %struct.S*
// UNTIEDRT: store %struct.S* [[S2_PTR]], %struct.S** [[S2_ADDR_PTR]],
// UNTIEDRT: load i32*, i32** %
// UNTIEDRT: store i32 2, i32* %
// UNTIEDRT: call i32 @__kmpc_omp_task(%
// UNTIEDRT: br label %[[EXIT]]

// UNTIEDRT: call void [[CONSTR]](%struct.S* {{[^,]*}} [[S2_ADDR]])
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(%
// CHECK: load i32*, i32** %
// CHECK: store i32 {{2|3}}, i32* %
// CHECK: call i32 @__kmpc_omp_task(%
// UNTIEDRT: br label %[[EXIT]]

// CHECK: call i32 @__kmpc_omp_taskyield(%
// CHECK: load i32*, i32** %
// CHECK: store i32 {{3|4}}, i32* %
// CHECK: call i32 @__kmpc_omp_task(%
// UNTIEDRT: br label %[[EXIT]]

// s1 = S();
// UNTIEDRT: call void [[CONSTR]](%struct.S* {{[^,]*}} [[TMP:%.+]])
// UNTIEDRT: [[DST:%.+]] = bitcast %struct.S* [[S1_ADDR]] to i8*
// UNTIEDRT: [[SRC:%.+]] = bitcast %struct.S* [[TMP]] to i8*
// UNTIEDRT: call void @llvm.memcpy.{{.+}}(i8* {{.*}}[[DST]], i8* {{.*}}[[SRC]], i64 4, i1 false)
// UNTIEDRT: call void [[DESTR:@.+]](%struct.S* {{[^,]*}} [[TMP]])

// CHECK: call i32 @__kmpc_omp_taskwait(%
// CHECK: load i32*, i32** %
// CHECK: store i32 {{4|5}}, i32* %
// CHECK: call i32 @__kmpc_omp_task(%
// UNTIEDRT: br label %[[EXIT]]

// UNTIEDRT: call void [[DESTR]](%struct.S* {{[^,]*}} [[S2_ADDR]])
// UNTIEDRT: [[S2_VOID_PTR:%.+]] = bitcast %struct.S* [[S2_ADDR]] to i8*
// UNTIEDRT: call void @__kmpc_free(i32 %{{.+}}, i8* [[S2_VOID_PTR]], i8* inttoptr (i64 7 to i8*))
// UNTIEDRT: call void [[DESTR]](%struct.S* {{[^,]*}} [[S1_ADDR]])
// CHECK: br label %[[CLEANUP]]

// CHECK: [[CLEANUP]]:
// UNTIEDRT: br label %[[EXIT]]

// UNTIEDRT:      [[EXIT]]:
// UNTIEDRT-NEXT: ret i32 0

struct S1 {
  int a;
  S1() { taskinit(); }
  void taskinit() {
#pragma omp task
    a = 0;
  }
} s1;

// CHECK-LABEL: taskinit
// CHECK: call i8* @__kmpc_omp_task_alloc(

#ifdef UNTIEDRT
// FIXME: There is a buffer overflow in IrBuilder mode.
template <typename T = void>
void foobar() {
  float a;
#pragma omp parallel
#pragma omp single
  {
    double b;
#pragma omp task
    a += b;
  }
}

// UNTIEDRT: define{{.*}} void @{{.+}}xxxx{{.+}}()
void xxxx() {
  // UNTIEDRT: call void @{{.+}}foobar{{.+}}()
  foobar();
}
// UNTIEDRT: define {{.*}}void @{{.+}}foobar{{.+}}()
// UNTIEDRT: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* {{.+}}, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float*)* [[PAR_OUTLINED:@.+]] to void (i32*, i32*, ...)*), float* %{{.+}})

// UNTIEDRT: define internal void [[PAR_OUTLINED]](i32* {{.+}}, i32* {{.+}}, float* {{.*}}[[A_ADDR:%.+]])
// UNTIEDRT: [[A_ADDR_REF:%.+]] = alloca float*,
// UNTIEDRT: [[B_ADDR:%.+]] = alloca double,
// UNTIEDRT: [[A_ADDR:%.+]] = load float*, float** [[A_ADDR_REF]],

// Copy `a` to the list of shared variables
// UNTIEDRT: [[SHARED_A:%.+]] = getelementptr inbounds %{{.+}}, [[SHAREDS_TY:%.+]]* [[SHAREDS:%.+]], i32 0, i32 0
// UNTIEDRT: store float* [[A_ADDR]], float** [[SHARED_A]],

// Allocate task.
// UNTIEDRT: [[RES:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* {{.+}}, i32 {{.+}}, i32 1, i64 48, i64 8, i32 (i32, i8*)* bitcast (i32 (i32, [[T_TASK_TY:%.+]]*)* @{{.+}} to i32 (i32, i8*)*))
// UNTIEDRT: [[TD:%.+]] = bitcast i8* [[RES]] to [[T_TASK_TY]]*
// Copy shared vars.
// UNTIEDRT: [[TD_TASK:%.+]] = getelementptr inbounds [[T_TASK_TY]], [[T_TASK_TY]]* [[TD]], i32 0, i32 0
// UNTIEDRT: [[TD_TASK_SHARES_REF:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[TD_TASK]], i32 0, i32 0
// UNTIEDRT: [[TD_TASK_SHARES:%.+]] = load i8*, i8** [[TD_TASK_SHARES_REF]],
// UNTIEDRT: [[SHAREDS_BC:%.+]] = bitcast [[SHAREDS_TY]]* [[SHAREDS]] to i8*
// UNTIEDRT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[TD_TASK_SHARES]], i8* align 8 [[SHAREDS_BC]], i64 8, i1 false)

// Copy firstprivate value of `b`.
// UNTIEDRT: [[TD_TASK_PRIVS:%.+]] = getelementptr inbounds [[T_TASK_TY]], [[T_TASK_TY]]* [[TD]], i32 0, i32 1
// UNTIEDRT: [[TD_TASK_PRIVS_B:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}}* [[TD_TASK_PRIVS]], i32 0, i32 0
// UNTIEDRT: [[B_VAL:%.+]] = load double, double* [[B_ADDR]],
// UNTIEDRT: store double [[B_VAL]], double* [[TD_TASK_PRIVS_B]],
#endif // UNTIEDRT
#endif
