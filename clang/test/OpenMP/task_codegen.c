// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=50 -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=50 -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef void *omp_depend_t;

void foo();

// CHECK-LABEL: @main
int main() {
  omp_depend_t d, x;
  int a;
  // CHECK: [[D_ADDR:%.+]] = alloca i8*,
  // CHECK: [[X_ADDR:%.+]] = alloca i8*,
  // CHECK: [[A_ADDR:%.+]] = alloca i32,
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
  // CHECK: [[ALLOC:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %0, i32 1, i64 40, i64 0, i32 (i32, i8*)* bitcast (i32 (i32, [[PRIVATES_TY:%.+]]*)* [[TASK_ENTRY:@.+]] to i32 (i32, i8*)*))
  // CHECK: [[DATA:%.+]] = bitcast i8* [[ALLOC]] to [[PRIVATES_TY]]*
  // CHECK: [[D:%.+]] = load i8*, i8** [[D_ADDR]],
  // CHECK: [[D_DEP:%.+]] = bitcast i8* [[D]] to %struct.kmp_depend_info*
  // CHECK: [[D_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[D_DEP]], i{{.+}} -1
  // CHECK: [[D_DEP_BASE_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[D_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE1:%.+]] = load i64, i64* [[D_DEP_BASE_SIZE]],
  // CHECK: [[SIZE:%.+]] = add nuw i64 0, [[SIZE1]]
  // CHECK: [[X:%.+]] = load i8*, i8** [[X_ADDR]],
  // CHECK: [[X_DEP:%.+]] = bitcast i8* [[X]] to %struct.kmp_depend_info*
  // CHECK: [[X_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[X_DEP]], i{{.+}} -1
  // CHECK: [[X_DEP_BASE_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[X_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE2:%.+]] = load i64, i64* [[X_DEP_BASE_SIZE]],
  // CHECK: [[SIZE3:%.+]] = add nuw i64 [[SIZE]], [[SIZE2]]
  // CHECK: [[SIZE:%.+]] = add nuw i64 [[SIZE3]], 1
  // CHECK: [[SIZE32:%.+]] = trunc i64 [[SIZE]] to i32
  // CHECK: [[SIZE64:%.+]] = zext i32 [[SIZE32]] to i64
  // CHECK: [[SV:%.+]] = call i8* @llvm.stacksave()
  // CHECK: store i8* [[SV]], i8** [[SV_ADDR:%.+]],
  // CHECK: [[VLA:%.+]] = alloca %struct.kmp_depend_info, i64 [[SIZE64]],
  // CHECK: [[VLA0:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 0
  // CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[A_ADDR_CAST:%.+]] = ptrtoint i32* [[A_ADDR]] to i64
  // CHECK: store i64 [[A_ADDR_CAST]], i64* [[BASE_ADDR]],
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 4, i64* [[SIZE_ADDR]],
  // CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, i8* [[FLAGS_ADDR]],
  // CHECK: [[VLA_D:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 1
  // CHECK: [[D_SIZE:%.+]] = mul nuw i64 24, [[SIZE1]]
  // CHECK: [[DEST:%.+]] = bitcast %struct.kmp_depend_info* [[VLA_D]] to i8*
  // CHECK: [[SRC:%.+]] = bitcast %struct.kmp_depend_info* [[D_DEP]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[DEST]], i8* align 8 [[SRC]], i64 [[D_SIZE]], i1 false)
  // CHECK: [[VLA_X:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* %25, i64 [[SIZE1]]
  // CHECK: [[X_SIZE:%.+]] = mul nuw i64 24, [[SIZE2]]
  // CHECK: [[DEST:%.+]] = bitcast %struct.kmp_depend_info* [[VLA_X]] to i8*
  // CHECK: [[SRC:%.+]] = bitcast %struct.kmp_depend_info* [[X_DEP]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 [[DEST]], i8* align 8 [[SRC]], i64 [[X_SIZE]], i1 false)
  // CHECK: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[VLA]] to i8*
  // CHECK: call i32 @__kmpc_omp_task_with_deps(%struct.ident_t* @{{.+}}, i32 [[GTID]], i8* [[ALLOC]], i32 [[SIZE32]], i8* [[BC]], i32 0, i8* null)
  // CHECK: [[SV:%.+]] = load i8*, i8** [[SV_ADDR]],
  // CHECK: call void @llvm.stackrestore(i8* [[SV]])
#pragma omp task depend(in: a) depend(depobj: d, x)
  {
#pragma omp taskgroup
    {
#pragma omp task
      foo();
    }
  }
  // CHECK: ret i32 0
  return 0;
}
// CHECK: call void @__kmpc_taskgroup(
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LINE: @bar
void bar() {
  // CHECK: call void @__kmpc_for_static_init_4(
#pragma omp for
for (int i = 0; i < 10; ++i)
  // CHECK: [[BUF:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 1, i64 48,
  // CHECK: [[BC_BUF:%.+]] = bitcast i8* [[BUF]] to [[TT_WITH_PRIVS:%.+]]*
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds [[TT_WITH_PRIVS]], [[TT_WITH_PRIVS]]* [[BC_BUF]], i32 0, i32 1
  // CHECK: [[I_PRIV:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}} [[PRIVS]], i32 0, i32 0
  // CHECK: store i32 %{{.+}}, i32* [[I_PRIV]],
  // CHECK: = call i32 @__kmpc_omp_task(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[BUF]])
#pragma omp task
++i;
}
#endif
