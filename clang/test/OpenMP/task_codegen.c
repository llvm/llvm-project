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
typedef __UINTPTR_TYPE__ omp_event_handle_t;

void foo(void);

// CHECK-LABEL: @main
int main(void) {
  omp_depend_t d, x;
  omp_event_handle_t evt;
  int a, *b;
  // CHECK: [[D_ADDR:%.+]] = alloca ptr,
  // CHECK: [[X_ADDR:%.+]] = alloca ptr,
  // CHECK: [[EVT_ADDR:%.+]] = alloca i64,
  // CHECK: [[A_ADDR:%.+]] = alloca i32,
  // CHECK: [[DEPOBJ_SIZE_ADDR:%.+]] = alloca i64,
  // CHECK: [[DEPOBJ_SIZE_ADDR1:%.+]] = alloca i64,
  // CHECK: = alloca i64,
  // CHECK: [[DEP_COUNTER_ADDR:%.+]] = alloca i64,
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
  // CHECK: [[ALLOC:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 [[GTID]], i32 65, i64 48, i64 0, ptr [[TASK_ENTRY:@.+]])
  // CHECK: [[EVT_VAL:%.+]] = call ptr @__kmpc_task_allow_completion_event(ptr @{{.+}}, i32 [[GTID]], ptr [[ALLOC]])
  // CHECK: [[CAST_EVT_VAL:%.+]] = ptrtoint ptr [[EVT_VAL]] to i64
  // CHECK: store i64 [[CAST_EVT_VAL]], ptr [[EVT_ADDR]], align 8
  // CHECK: [[D_DEP:%.+]] = load ptr, ptr [[D_ADDR]], align 8
  // CHECK: [[D_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[D_DEP]], i{{.+}} -1
  // CHECK: [[D_DEP_BASE_SIZE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[D_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE1:%.+]] = load i64, ptr [[D_DEP_BASE_SIZE]], align 8
  // CHECK-DAG: store i64 0, ptr [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SZ:%.+]] = load i64, ptr [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SIZE:%.+]] = add nuw i64 [[SZ]], [[SIZE1]]
  // CHECK: store i64 [[SIZE]], ptr [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[X_DEP:%.+]] = load ptr, ptr [[X_ADDR]], align 8
  // CHECK: [[X_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[X_DEP]], i{{.+}} -1
  // CHECK: [[X_DEP_BASE_SIZE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[X_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE2:%.+]] = load i64, ptr [[X_DEP_BASE_SIZE]], align 8
  // CHECK-DAG: store i64 0, ptr [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SZ:%.+]] = load i64, ptr [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SIZE3:%.+]] = add nuw i64 [[SZ]], [[SIZE2]]
  // CHECK: store i64 [[SIZE3]], ptr [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SZ:%.+]] = load i64, ptr [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SZ1:%.+]] = load i64, ptr [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SIZE1:%.+]] = add nuw i64 0, [[SZ]]
  // CHECK: [[SIZE2:%.+]] = add nuw i64 [[SIZE1]], [[SZ1]]
  // CHECK: [[SIZE:%.+]] = add nuw i64 [[SIZE2]], 2
  // CHECK: [[SV:%.+]] = call ptr @llvm.stacksave.p0()
  // CHECK: store ptr [[SV]], ptr [[SV_ADDR:%.+]], align 8
  // CHECK: [[VLA:%.+]] = alloca %struct.kmp_depend_info, i64 [[SIZE]],
  // CHECK: [[SIZE32:%.+]] = trunc i64 [[SIZE]] to i32
  // CHECK: [[A_ADDR_CAST:%.+]] = ptrtoint ptr [[A_ADDR]] to i64
  // CHECK: [[VLA0:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[VLA]], i64 0
  // CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA0]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[A_ADDR_CAST]], ptr [[BASE_ADDR]], align 16
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA0]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 4, ptr [[SIZE_ADDR]], align 8
  // CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA0]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, ptr [[FLAGS_ADDR]], align 1
  // CHECK: [[A:%.+]] = load i32, ptr [[A_ADDR]], align 4
  // CHECK: [[A_CAST:%.+]] = sext i32 [[A]] to i64
  // CHECK: [[SZ1:%.+]] = mul nuw i64 24, [[A_CAST]]
  // CHECK: [[A:%.+]] = load i32, ptr [[A_ADDR]], align 4
  // CHECK: [[A_CAST:%.+]] = sext i32 [[A]] to i64
  // CHECK: [[SZ:%.+]] = mul nuw i64 [[SZ1]], [[A_CAST]]
  // CHECK: [[B_ADDR_CAST:%.+]] = ptrtoint ptr %{{.+}} to i64
  // CHECK: [[VLA1:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[VLA]], i64 1
  // CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA1]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[B_ADDR_CAST]], ptr [[BASE_ADDR]], align 8
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA1]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 [[SZ]], ptr [[SIZE_ADDR]], align 8
  // CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[VLA1]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, ptr [[FLAGS_ADDR]], align 8
  // CHECK: store i64 2, ptr [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[BC:%.+]] = load ptr, ptr [[D_ADDR]], align 8
  // CHECK: [[PREV:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[BC]], i64 -1
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[PREV]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE:%.+]] = load i64, ptr [[SIZE_ADDR]], align 8
  // CHECK: [[BYTES:%.+]] = mul nuw i64 24, [[SIZE]]
  // CHECK: [[POS:%.+]] = load i64, ptr [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[VLA_D:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[VLA]], i64 [[POS]]
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align {{.+}} [[VLA_D]], ptr align {{.+}} [[BC]], i64 [[BYTES]], i1 false)
  // CHECK: [[ADD:%.+]] = add nuw i64 [[POS]], [[SIZE]]
  // CHECK: store i64 [[ADD]], ptr [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[BC:%.+]] = load ptr, ptr [[X_ADDR]], align 8
  // CHECK: [[PREV:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[BC]], i64 -1
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[PREV]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE:%.+]] = load i64, ptr [[SIZE_ADDR]], align 8
  // CHECK: [[BYTES:%.+]] = mul nuw i64 24, [[SIZE]]
  // CHECK: [[POS:%.+]] = load i64, ptr [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[VLA_X:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[VLA]], i64 [[POS]]
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align {{.+}} [[VLA_X]], ptr align {{.+}} [[BC]], i64 [[BYTES]], i1 false)
  // CHECK: [[ADD:%.+]] = add nuw i64 [[POS]], [[SIZE]]
  // CHECK: store i64 [[ADD]], ptr [[DEP_COUNTER_ADDR]], align 8
  // CHECK: call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 [[GTID]], ptr [[ALLOC]], i32 [[SIZE32]], ptr [[VLA]], i32 0, ptr null)
  // CHECK: [[SV:%.+]] = load ptr, ptr [[SV_ADDR]], align 8
  // CHECK: call void @llvm.stackrestore.p0(ptr [[SV]])
#pragma omp task depend(in: a, ([3][a][a])&b) depend(depobj: d, x) detach(evt)
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
// CHECK: call ptr @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LINE: @bar
void bar(void) {
  int **a;
  // CHECK: call void @__kmpc_for_static_init_4(
#pragma omp for
for (int i = 0; i < 10; ++i)
  // CHECK: [[BUF:%.+]] = call ptr @__kmpc_omp_task_alloc(ptr @{{.+}}, i32 %{{.+}}, i32 1, i64 48,
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds nuw [[TT_WITH_PRIVS:%.+]], ptr [[BUF]], i32 0, i32 1
  // CHECK: [[I_PRIV:%.+]] = getelementptr inbounds nuw %{{.+}}, ptr [[PRIVS]], i32 0, i32 0
  // CHECK: [[I:%.+]] = load i32, ptr [[I_ADDR:%.+]],
  // CHECK: store i32 %{{.+}}, ptr [[I_PRIV]],

  // NELEMS = 1 * ((i - 0 + 2 - 1) / 2);
  // CHECK: [[END:%.+]] = load i32, ptr [[I_ADDR]],
  // CHECK: [[EB_SUB:%.+]] = sub i32 [[END]], 0
  // CHECK: [[EB_SUB_2_ADD:%.+]] = add i32 [[EB_SUB]], 2
  // CHECK: [[EB_SUB_2_ADD_1_SUB:%.+]] = sub i32 [[EB_SUB_2_ADD]], 1
  // CHECK: [[EB_SUB_2_ADD_1_SUB_2_DIV:%.+]] = udiv i32 [[EB_SUB_2_ADD_1_SUB]], 2
  // CHECK: [[ELEMS:%.+]] = zext i32 [[EB_SUB_2_ADD_1_SUB_2_DIV]] to i64
  // CHECK: [[ELEMS2:%.+]] = mul nuw i64 [[ELEMS]], 1
  // CHECK: [[NELEMS:%.+]] = mul nuw i64 [[ELEMS2]], 1

  // ITERATOR_TOTAL = NELEMS + 0;
  // CHECK: [[ITERATOR_TOTAL:%.+]] = add nuw i64 0, [[NELEMS]]
  // NELEMS = ITERATOR_TOTAL + non-iterator-deps (=0)
  // CHECK: [[TOTAL:%.+]] = add nuw i64 [[ITERATOR_TOTAL]], 0

  // %struct.kmp_depend_info DEPS[TOTAL];
  // CHECK: [[DEPS:%.+]] = alloca %struct.kmp_depend_info, i64 [[TOTAL]],
  // CHECK: [[NDEPS:%.+]] = trunc i64 [[TOTAL]] to i32

  // i64 DEP_COUNTER = 0;
  // CHECK: store i64 0, ptr [[DEP_COUNTER_ADDR:%.+]],

  // NELEMS = ((i - 0 + 2 - 1) / 2);
  // CHECK: [[END:%.+]] = load i32, ptr [[I_ADDR]],
  // CHECK: [[EB_SUB:%.+]] = sub i32 [[END]], 0
  // CHECK: [[EB_SUB_2_ADD:%.+]] = add i32 [[EB_SUB]], 2
  // CHECK: [[EB_SUB_2_ADD_1_SUB:%.+]] = sub i32 [[EB_SUB_2_ADD]], 1
  // CHECK: [[ELEMS:%.+]] = udiv i32 [[EB_SUB_2_ADD_1_SUB]], 2

  // i32 COUNTER = 0;
  // CHECK: store i32 0, ptr [[COUNTER_ADDR:%.+]],
  // CHECK: br label %[[CONT:.+]]

  // Loop.
  // CHECK: [[CONT]]:
  // CHECK: [[COUNTER:%.+]] = load i32, ptr [[COUNTER_ADDR]],
  // CHECK: [[CMP:%.+]] = icmp ult i32 [[COUNTER]], [[ELEMS]]
  // CHECK: br i1 [[CMP]], label %[[BODY:.+]], label %[[EXIT:.+]]

  // CHECK: [[BODY]]:

  // k = 0 + 2*COUNTER;
  // CHECK: [[COUNTER:%.+]] = load i32, ptr [[COUNTER_ADDR]],
  // CHECK: [[C2_MUL:%.+]] = mul i32 [[COUNTER]], 2
  // CHECK: [[C2_MUL_0_ADD:%.+]] = add i32 0, [[C2_MUL]]
  // CHECK: store i32 [[C2_MUL_0_ADD]], ptr [[K_ADDR:%.+]],

  // &a[k][i]
  // CHECK: [[A:%.+]] = load ptr, ptr [[A_ADDR:%.+]],
  // CHECK: [[K:%.+]] = load i32, ptr [[K_ADDR]],
  // CHECK: [[IDX:%.+]] = zext i32 [[K]] to i64
  // CHECK: [[AK_ADDR:%.+]] = getelementptr inbounds ptr, ptr [[A]], i64 [[IDX]]
  // CHECK: [[AK:%.+]] = load ptr, ptr [[AK_ADDR]],
  // CHECK: [[I:%.+]] = load i32, ptr [[I_ADDR]],
  // CHECK: [[IDX:%.+]] = sext i32 [[I]] to i64
  // CHECK: [[AKI_ADDR:%.+]] = getelementptr inbounds i32, ptr [[AK]], i64 [[IDX]]
  // CHECK: [[AKI_INT:%.+]] = ptrtoint ptr [[AKI_ADDR]] to i64

  // DEPS[DEP_COUNTER].base_addr = &a[k][i];
  // CHECK: [[DEP_COUNTER:%.+]] = load i64, ptr [[DEP_COUNTER_ADDR]],
  // CHECK: [[DEPS_DC:%.+]] = getelementptr %struct.kmp_depend_info, ptr [[DEPS]], i64 [[DEP_COUNTER]]
  // CHECK: [[DEPS_DC_BASE_ADDR:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEPS_DC]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[AKI_INT]], ptr [[DEPS_DC_BASE_ADDR]],

  // DEPS[DEP_COUNTER].size = sizeof(a[k][i]);
  // CHECK: [[DEPS_DC_SIZE:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEPS_DC]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 4, ptr [[DEPS_DC_SIZE]],

  // DEPS[DEP_COUNTER].flags = in;
  // CHECK: [[DEPS_DC_FLAGS:%.+]] = getelementptr inbounds nuw %struct.kmp_depend_info, ptr [[DEPS_DC]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, ptr [[DEPS_DC_FLAGS]],

  // DEP_COUNTER = DEP_COUNTER + 1;
  // CHECK: [[DEP_COUNTER:%.+]] = load i64, ptr [[DEP_COUNTER_ADDR]],
  // CHECK: [[INC:%.+]] = add nuw i64 [[DEP_COUNTER]], 1
  // CHECK: store i64 [[INC]], ptr [[DEP_COUNTER_ADDR]],

  // COUNTER = COUNTER + 1;
  // CHECK: [[COUNTER:%.+]] = load i32, ptr [[COUNTER_ADDR]],
  // CHECK: [[INC:%.+]] = add i32 [[COUNTER]], 1
  // CHECK: store i32 [[INC]], ptr [[COUNTER_ADDR]],
  // CHECK: br label %[[CONT]]

  // CHECK: [[EXIT]]:
  // CHECK: = call i32 @__kmpc_omp_task_with_deps(ptr @{{.+}}, i32 %{{.+}}, ptr [[BUF]], i32 [[NDEPS]], ptr [[DEPS]], i32 0, ptr null)
#pragma omp task depend(iterator(unsigned k=0:i:2), in: a[k][i])
++i;
}
#endif
