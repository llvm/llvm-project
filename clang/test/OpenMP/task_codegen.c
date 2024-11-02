// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp -fopenmp-version=50 -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -no-opaque-pointers -verify -triple x86_64-apple-darwin10 -fopenmp-simd -fopenmp-version=50 -x c -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -x c -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
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
  // CHECK: [[D_ADDR:%.+]] = alloca i8*,
  // CHECK: [[X_ADDR:%.+]] = alloca i8*,
  // CHECK: [[EVT_ADDR:%.+]] = alloca i64,
  // CHECK: [[A_ADDR:%.+]] = alloca i32,
  // CHECK: [[DEPOBJ_SIZE_ADDR:%.+]] = alloca i64,
  // CHECK: [[DEPOBJ_SIZE_ADDR1:%.+]] = alloca i64,
  // CHECK: = alloca i64,
  // CHECK: [[DEP_COUNTER_ADDR:%.+]] = alloca i64,
  // CHECK: [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(
  // CHECK: [[ALLOC:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 [[GTID]], i32 65, i64 48, i64 0, i32 (i32, i8*)* bitcast (i32 (i32, [[PRIVATES_TY:%.+]]*)* [[TASK_ENTRY:@.+]] to i32 (i32, i8*)*))
  // CHECK: [[EVT_VAL:%.+]] = call i8* @__kmpc_task_allow_completion_event(%struct.ident_t* @{{.+}}, i32 [[GTID]], i8* [[ALLOC]])
  // CHECK: [[CAST_EVT_VAL:%.+]] = ptrtoint i8* [[EVT_VAL]] to i64
  // CHECK: store i64 [[CAST_EVT_VAL]], i64* [[EVT_ADDR]], align 8
  // CHECK: [[DATA:%.+]] = bitcast i8* [[ALLOC]] to [[PRIVATES_TY]]*
  // CHECK: [[D_ADDR_CAST:%.+]] = bitcast i8** [[D_ADDR]] to %struct.kmp_depend_info**
  // CHECK: [[D_DEP:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[D_ADDR_CAST]], align 8
  // CHECK: [[D_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[D_DEP]], i{{.+}} -1
  // CHECK: [[D_DEP_BASE_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[D_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE1:%.+]] = load i64, i64* [[D_DEP_BASE_SIZE]], align 8
  // CHECK-DAG: store i64 0, i64* [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SZ:%.+]] = load i64, i64* [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SIZE:%.+]] = add nuw i64 [[SZ]], [[SIZE1]]
  // CHECK: store i64 [[SIZE]], i64* [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[X_ADDR_CAST:%.+]] = bitcast i8** [[X_ADDR]] to %struct.kmp_depend_info**
  // CHECK: [[X_DEP:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[X_ADDR_CAST]], align 8
  // CHECK: [[X_DEP_BASE:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[X_DEP]], i{{.+}} -1
  // CHECK: [[X_DEP_BASE_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[X_DEP_BASE]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE2:%.+]] = load i64, i64* [[X_DEP_BASE_SIZE]], align 8
  // CHECK-DAG: store i64 0, i64* [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SZ:%.+]] = load i64, i64* [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SIZE3:%.+]] = add nuw i64 [[SZ]], [[SIZE2]]
  // CHECK: store i64 [[SIZE3]], i64* [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SZ:%.+]] = load i64, i64* [[DEPOBJ_SIZE_ADDR]], align 8
  // CHECK: [[SZ1:%.+]] = load i64, i64* [[DEPOBJ_SIZE_ADDR1]], align 8
  // CHECK: [[SIZE1:%.+]] = add nuw i64 0, [[SZ]]
  // CHECK: [[SIZE2:%.+]] = add nuw i64 [[SIZE1]], [[SZ1]]
  // CHECK: [[SIZE:%.+]] = add nuw i64 [[SIZE2]], 2
  // CHECK: [[SV:%.+]] = call i8* @llvm.stacksave()
  // CHECK: store i8* [[SV]], i8** [[SV_ADDR:%.+]], align 8
  // CHECK: [[VLA:%.+]] = alloca %struct.kmp_depend_info, i64 [[SIZE]],
  // CHECK: [[SIZE32:%.+]] = trunc i64 [[SIZE]] to i32
  // CHECK: [[A_ADDR_CAST:%.+]] = ptrtoint i32* [[A_ADDR]] to i64
  // CHECK: [[VLA0:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 0
  // CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[A_ADDR_CAST]], i64* [[BASE_ADDR]], align 16
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 4, i64* [[SIZE_ADDR]], align 8
  // CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA0]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, i8* [[FLAGS_ADDR]], align 1
  // CHECK: [[A:%.+]] = load i32, i32* [[A_ADDR]], align 4
  // CHECK: [[A_CAST:%.+]] = sext i32 [[A]] to i64
  // CHECK: [[SZ1:%.+]] = mul nuw i64 24, [[A_CAST]]
  // CHECK: [[A:%.+]] = load i32, i32* [[A_ADDR]], align 4
  // CHECK: [[A_CAST:%.+]] = sext i32 [[A]] to i64
  // CHECK: [[SZ:%.+]] = mul nuw i64 [[SZ1]], [[A_CAST]]
  // CHECK: [[B_ADDR_CAST:%.+]] = ptrtoint i32** %{{.+}} to i64
  // CHECK: [[VLA1:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 1
  // CHECK: [[BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA1]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[B_ADDR_CAST]], i64* [[BASE_ADDR]], align 8
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA1]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 [[SZ]], i64* [[SIZE_ADDR]], align 8
  // CHECK: [[FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA1]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, i8* [[FLAGS_ADDR]], align 8
  // CHECK: store i64 2, i64* [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[D_ADDR_CAST:%.+]] = bitcast i8** [[D_ADDR]] to %struct.kmp_depend_info**
  // CHECK: [[BC:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[D_ADDR_CAST]], align 8
  // CHECK: [[PREV:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[BC]], i64 -1
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[PREV]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE:%.+]] = load i64, i64* [[SIZE_ADDR]], align 8
  // CHECK: [[BYTES:%.+]] = mul nuw i64 24, [[SIZE]]
  // CHECK: [[POS:%.+]] = load i64, i64* [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[VLA_D:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 [[POS]]
  // CHECK: [[DEST:%.+]] = bitcast %struct.kmp_depend_info* [[VLA_D]] to i8*
  // CHECK: [[SRC:%.+]] = bitcast %struct.kmp_depend_info* [[BC]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {{.+}} [[DEST]], i8* align {{.+}} [[SRC]], i64 [[BYTES]], i1 false)
  // CHECK: [[ADD:%.+]] = add nuw i64 [[POS]], [[SIZE]]
  // CHECK: store i64 [[ADD]], i64* [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[X_ADDR_CAST:%.+]] = bitcast i8** [[X_ADDR]] to %struct.kmp_depend_info**
  // CHECK: [[BC:%.+]] = load %struct.kmp_depend_info*, %struct.kmp_depend_info** [[X_ADDR_CAST]], align 8
  // CHECK: [[PREV:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[BC]], i64 -1
  // CHECK: [[SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[PREV]], i{{.+}} 0, i{{.+}} 0
  // CHECK: [[SIZE:%.+]] = load i64, i64* [[SIZE_ADDR]], align 8
  // CHECK: [[BYTES:%.+]] = mul nuw i64 24, [[SIZE]]
  // CHECK: [[POS:%.+]] = load i64, i64* [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[VLA_X:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[VLA]], i64 [[POS]]
  // CHECK: [[DEST:%.+]] = bitcast %struct.kmp_depend_info* [[VLA_X]] to i8*
  // CHECK: [[SRC:%.+]] = bitcast %struct.kmp_depend_info* [[BC]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align {{.+}} [[DEST]], i8* align {{.+}} [[SRC]], i64 [[BYTES]], i1 false)
  // CHECK: [[ADD:%.+]] = add nuw i64 [[POS]], [[SIZE]]
  // CHECK: store i64 [[ADD]], i64* [[DEP_COUNTER_ADDR]], align 8
  // CHECK: [[BC:%.+]] = bitcast %struct.kmp_depend_info* [[VLA]] to i8*
  // CHECK: call i32 @__kmpc_omp_task_with_deps(%struct.ident_t* @{{.+}}, i32 [[GTID]], i8* [[ALLOC]], i32 [[SIZE32]], i8* [[BC]], i32 0, i8* null)
  // CHECK: [[SV:%.+]] = load i8*, i8** [[SV_ADDR]], align 8
  // CHECK: call void @llvm.stackrestore(i8* [[SV]])
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
// CHECK: call i8* @__kmpc_omp_task_alloc(
// CHECK: call i32 @__kmpc_omp_task(
// CHECK: call void @__kmpc_end_taskgroup(

// CHECK-LINE: @bar
void bar(void) {
  int **a;
  // CHECK: call void @__kmpc_for_static_init_4(
#pragma omp for
for (int i = 0; i < 10; ++i)
  // CHECK: [[BUF:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 1, i64 48,
  // CHECK: [[BC_BUF:%.+]] = bitcast i8* [[BUF]] to [[TT_WITH_PRIVS:%.+]]*
  // CHECK: [[PRIVS:%.+]] = getelementptr inbounds [[TT_WITH_PRIVS]], [[TT_WITH_PRIVS]]* [[BC_BUF]], i32 0, i32 1
  // CHECK: [[I_PRIV:%.+]] = getelementptr inbounds %{{.+}}, %{{.+}} [[PRIVS]], i32 0, i32 0
  // CHECK: [[I:%.+]] = load i32, i32* [[I_ADDR:%.+]],
  // CHECK: store i32 %{{.+}}, i32* [[I_PRIV]],

  // NELEMS = 1 * ((i - 0 + 2 - 1) / 2);
  // CHECK: [[END:%.+]] = load i32, i32* [[I_ADDR]],
  // CHECK: [[EB_SUB:%.+]] = sub i32 [[END]], 0
  // CHECK: [[EB_SUB_2_ADD:%.+]] = add i32 [[EB_SUB]], 2
  // CHECK: [[EB_SUB_2_ADD_1_SUB:%.+]] = sub i32 [[EB_SUB_2_ADD]], 1
  // CHECK: [[EB_SUB_2_ADD_1_SUB_2_DIV:%.+]] = udiv i32 [[EB_SUB_2_ADD_1_SUB]], 2
  // CHECK: [[ELEMS:%.+]] = zext i32 [[EB_SUB_2_ADD_1_SUB_2_DIV]] to i64
  // CHECK: [[NELEMS:%.+]] = mul nuw i64 [[ELEMS]], 1

  // ITERATOR_TOTAL = NELEMS + 0;
  // CHECK: [[ITERATOR_TOTAL:%.+]] = add nuw i64 0, [[NELEMS]]
  // NELEMS = ITERATOR_TOTAL + non-iterator-deps (=0)
  // CHECK: [[TOTAL:%.+]] = add nuw i64 [[ITERATOR_TOTAL]], 0

  // %struct.kmp_depend_info DEPS[TOTAL];
  // CHECK: [[DEPS:%.+]] = alloca %struct.kmp_depend_info, i64 [[TOTAL]],
  // CHECK: [[NDEPS:%.+]] = trunc i64 [[TOTAL]] to i32

  // i64 DEP_COUNTER = 0;
  // CHECK: store i64 0, i64* [[DEP_COUNTER_ADDR:%.+]],

  // NELEMS = ((i - 0 + 2 - 1) / 2);
  // CHECK: [[END:%.+]] = load i32, i32* [[I_ADDR]],
  // CHECK: [[EB_SUB:%.+]] = sub i32 [[END]], 0
  // CHECK: [[EB_SUB_2_ADD:%.+]] = add i32 [[EB_SUB]], 2
  // CHECK: [[EB_SUB_2_ADD_1_SUB:%.+]] = sub i32 [[EB_SUB_2_ADD]], 1
  // CHECK: [[ELEMS:%.+]] = udiv i32 [[EB_SUB_2_ADD_1_SUB]], 2

  // i32 COUNTER = 0;
  // CHECK: store i32 0, i32* [[COUNTER_ADDR:%.+]],
  // CHECK: br label %[[CONT:.+]]

  // Loop.
  // CHECK: [[CONT]]:
  // CHECK: [[COUNTER:%.+]] = load i32, i32* [[COUNTER_ADDR]],
  // CHECK: [[CMP:%.+]] = icmp ult i32 [[COUNTER]], [[ELEMS]]
  // CHECK: br i1 [[CMP]], label %[[BODY:.+]], label %[[EXIT:.+]]

  // CHECK: [[BODY]]:

  // k = 0 + 2*COUNTER;
  // CHECK: [[COUNTER:%.+]] = load i32, i32* [[COUNTER_ADDR]],
  // CHECK: [[C2_MUL:%.+]] = mul i32 [[COUNTER]], 2
  // CHECK: [[C2_MUL_0_ADD:%.+]] = add i32 0, [[C2_MUL]]
  // CHECK: store i32 [[C2_MUL_0_ADD]], i32* [[K_ADDR:%.+]],

  // &a[k][i]
  // CHECK: [[A:%.+]] = load i32**, i32*** [[A_ADDR:%.+]],
  // CHECK: [[K:%.+]] = load i32, i32* [[K_ADDR]],
  // CHECK: [[IDX:%.+]] = zext i32 [[K]] to i64
  // CHECK: [[AK_ADDR:%.+]] = getelementptr inbounds i32*, i32** [[A]], i64 [[IDX]]
  // CHECK: [[AK:%.+]] = load i32*, i32** [[AK_ADDR]],
  // CHECK: [[I:%.+]] = load i32, i32* [[I_ADDR]],
  // CHECK: [[IDX:%.+]] = sext i32 [[I]] to i64
  // CHECK: [[AKI_ADDR:%.+]] = getelementptr inbounds i32, i32* [[AK]], i64 [[IDX]]
  // CHECK: [[AKI_INT:%.+]] = ptrtoint i32* [[AKI_ADDR]] to i64

  // DEPS[DEP_COUNTER].base_addr = &a[k][i];
  // CHECK: [[DEP_COUNTER:%.+]] = load i64, i64* [[DEP_COUNTER_ADDR]],
  // CHECK: [[DEPS_DC:%.+]] = getelementptr %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEPS]], i64 [[DEP_COUNTER]]
  // CHECK: [[DEPS_DC_BASE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEPS_DC]], i{{.+}} 0, i{{.+}} 0
  // CHECK: store i64 [[AKI_INT]], i64* [[DEPS_DC_BASE_ADDR]],

  // DEPS[DEP_COUNTER].size = sizeof(a[k][i]);
  // CHECK: [[DEPS_DC_SIZE:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEPS_DC]], i{{.+}} 0, i{{.+}} 1
  // CHECK: store i64 4, i64* [[DEPS_DC_SIZE]],

  // DEPS[DEP_COUNTER].flags = in;
  // CHECK: [[DEPS_DC_FLAGS:%.+]] = getelementptr inbounds %struct.kmp_depend_info, %struct.kmp_depend_info* [[DEPS_DC]], i{{.+}} 0, i{{.+}} 2
  // CHECK: store i8 1, i8* [[DEPS_DC_FLAGS]],

  // DEP_COUNTER = DEP_COUNTER + 1;
  // CHECK: [[DEP_COUNTER:%.+]] = load i64, i64* [[DEP_COUNTER_ADDR]],
  // CHECK: [[INC:%.+]] = add nuw i64 [[DEP_COUNTER]], 1
  // CHECK: store i64 [[INC]], i64* [[DEP_COUNTER_ADDR]],

  // COUNTER = COUNTER + 1;
  // CHECK: [[COUNTER:%.+]] = load i32, i32* [[COUNTER_ADDR]],
  // CHECK: [[INC:%.+]] = add i32 [[COUNTER]], 1
  // CHECK: store i32 [[INC]], i32* [[COUNTER_ADDR]],
  // CHECK: br label %[[CONT]]

  // CHECK: [[EXIT]]:
  // CHECK: [[DEP_BEGIN:%.+]] = bitcast %struct.kmp_depend_info* [[DEPS]] to i8*
  // CHECK: = call i32 @__kmpc_omp_task_with_deps(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[BUF]], i32 [[NDEPS]], i8* [[DEP_BEGIN]], i32 0, i8* null)
#pragma omp task depend(iterator(unsigned k=0:i:2), in: a[k][i])
++i;
}
#endif
