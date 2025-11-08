// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int a[10];
// CIR: cir.global external @a = #cir.zero : !cir.array<!s32i x 10>

// LLVM: @a = global [10 x i32] zeroinitializer

// OGCG: @a = global [10 x i32] zeroinitializer

int aa[10][5];
// CIR: cir.global external @aa = #cir.zero : !cir.array<!cir.array<!s32i x 5> x 10>

// LLVM: @aa = global [10 x [5 x i32]] zeroinitializer

// OGCG: @aa = global [10 x [5 x i32]] zeroinitializer

int c[10] = {};
// CIR: cir.global external @c = #cir.zero : !cir.array<!s32i x 10>

// LLVM: @c = global [10 x i32] zeroinitializer

// OGCG: @c = global [10 x i32] zeroinitializer

int d[3] = {1, 2, 3};
// CIR: cir.global external @d = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>

// LLVM: @d = global [3 x i32] [i32 1, i32 2, i32 3]

// OGCG: @d = global [3 x i32] [i32 1, i32 2, i32 3]

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CIR: cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 3>

// LLVM: @dd = global [3 x [2 x i32]] [
// LLVM: [2 x i32] [i32 1, i32 2], [2 x i32]
// LLVM: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

// OGCG: @dd = global [3 x [2 x i32]] [
// OGCG: [2 x i32] [i32 1, i32 2], [2 x i32]
// OGCG: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

int e[10] = {1, 2};
// CIR: cir.global external @e = #cir.const_record<{#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.zero : !cir.array<!s32i x 8>}> : !rec_anon_struct

// LLVM: @e = global <{ i32, i32, [8 x i32] }> <{ i32 1, i32 2, [8 x i32] zeroinitializer }>

// OGCG: @e = global <{ i32, i32, [8 x i32] }> <{ i32 1, i32 2, [8 x i32] zeroinitializer }>

int f[5] = {1, 2};
// CIR: cir.global external @f = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 5>

// LLVM: @f = global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

// OGCG: @f = global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

int g[16] = {1, 2, 3, 4, 5, 6, 7, 8};
// CIR:      cir.global external @g = #cir.const_record<{
// CIR-SAME:   #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i,
// CIR-SAME:                     #cir.int<3> : !s32i, #cir.int<4> : !s32i,
// CIR-SAME:                     #cir.int<5> : !s32i, #cir.int<6> : !s32i,
// CIR-SAME:                     #cir.int<7> : !s32i, #cir.int<8> : !s32i]>
// CIR-SAME:                     : !cir.array<!s32i x 8>,
// CIR-SAME:   #cir.zero : !cir.array<!s32i x 8>}> : !rec_anon_struct1

// LLVM:       @g = global <{ [8 x i32], [8 x i32] }> 
// LLVM-SAME:          <{ [8 x i32]
// LLVM-SAME:              [i32 1, i32 2, i32 3, i32 4,
// LLVM-SAME:               i32 5, i32 6, i32 7, i32 8],
// LLVM-SAME:             [8 x i32] zeroinitializer }>

// OGCG:       @g = global <{ [8 x i32], [8 x i32] }> 
// OGCG-SAME:          <{ [8 x i32]
// OGCG-SAME:              [i32 1, i32 2, i32 3, i32 4,
// OGCG-SAME:               i32 5, i32 6, i32 7, i32 8],
// OGCG-SAME:             [8 x i32] zeroinitializer }>


extern int b[10];
// CIR: cir.global "private" external @b : !cir.array<!s32i x 10>
// LLVM: @b = external global [10 x i32]
// OGCG: @b = external global [10 x i32]

extern int bb[10][5];
// CIR: cir.global "private" external @bb : !cir.array<!cir.array<!s32i x 5> x 10>
// LLVM: @bb = external global [10 x [5 x i32]]
// OGCG: @bb = external global [10 x [5 x i32]]

// This function is only here to make sure the external globals are emitted.
void reference_externs() {
  b;
  bb;
}

// OGCG: @[[FUN2_ARR:.*]] = private unnamed_addr constant [2 x i32] [i32 5, i32 0], align 4
// OGCG: @[[FUN3_ARR:.*]] = private unnamed_addr constant [2 x i32] [i32 5, i32 6], align 4
// OGCG: @[[FUN4_ARR:.*]] = private unnamed_addr constant [2 x [1 x i32]] [
// OGCG: [1 x i32] [i32 5], [1 x i32] [i32 6]], align 4
// OGCG: @[[FUN5_ARR:.*]] = private unnamed_addr constant [2 x [1 x i32]] [
// OGCG: [1 x i32] [i32 5], [1 x i32] zeroinitializer], align 4

void func() {
  int arr[10];
  int e = arr[0];
  int e2 = arr[1];
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["arr"]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[INIT_2:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e2", init]
// CIR: %[[IDX:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR" cir.store %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>> -> !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR" cir.store %[[TMP]], %[[INIT_2]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z4funcv(){{.*}}
// LLVM-NEXT: %[[ARR:.*]] = alloca [10 x i32], i64 1, align 16
// LLVM-NEXT: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT: %[[INIT_2:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT: %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM-NEXT: %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 0
// LLVM-NEXT: %[[TMP_1:.*]] = load i32, ptr %[[ELE_PTR]], align 16
// LLVM-NEXT: store i32 %[[TMP_1]], ptr %[[INIT]], align 4
// LLVM-NEXT: %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM-NEXT: %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// LLVM-NEXT: %[[TMP_2:.*]] = load i32, ptr %[[ELE_PTR]], align 4
// LLVM-NEXT: store i32 %[[TMP_2]], ptr %[[INIT_2]], align 4

// OGCG: %[[ARR:.*]] = alloca [10 x i32], align 16
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: %[[INIT_2:.*]] = alloca i32, align 4
// OGCG: %[[ELE_PTR:.*]] = getelementptr inbounds [10 x i32], ptr %[[ARR]], i64 0, i64 0
// OGCG: %[[TMP_1:.*]] = load i32, ptr %[[ELE_PTR]], align 16
// OGCG: store i32 %[[TMP_1]], ptr %[[INIT]], align 4
// OGCG: %[[ELE_PTR:.*]] = getelementptr inbounds [10 x i32], ptr %[[ARR]], i64 0, i64 1
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[ELE_PTR]], align 4
// OGCG: store i32 %[[TMP_2]], ptr %[[INIT_2]], align 4

void func2() {
  int arr[2] = {5};
}

// CIR: %[[ARR2:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
// CIR: %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init]
// CIR: %[[ARR_0:.*]] = cir.cast array_to_ptrdecay %[[ARR2]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store{{.*}} %[[FIVE]], %[[ARR_0]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride %[[ARR_0]], %[[OFFSET_0]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.store{{.*}} %[[ELE_PTR]], %[[ARR_PTR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[ARR_END:.*]] = cir.ptr_stride %[[ARR_0]], %[[TWO]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: cir.do {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[ARR_CUR]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %[[ARR_NEXT]], %[[ARR_PTR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:   %[[CMP:.*]] = cir.cmp(ne, %[[ARR_CUR]], %[[ARR_END]]) : !cir.ptr<!s32i>, !cir.bool
// CIR:   cir.condition(%[[CMP]])
// CIR: }

// LLVM: define{{.*}} void @_Z5func2v(){{.*}}
// LLVM:   %[[ARR:.*]] = alloca [2 x i32], i64 1, align 4
// LLVM:   %[[TMP:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:   store i32 5, ptr %[[ARR_PTR]], align 4
// LLVM:   %[[ELE_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// LLVM:   store ptr %[[ELE_1_PTR]], ptr %[[TMP]], align 8
// LLVM:   %[[END_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 2
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_NEXT:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[END_PTR]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   store i32 0, ptr %[[CUR]], align 4
// LLVM:   %[[NEXT:.*]] = getelementptr i32, ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[TMP]], align 8
// LLVM:   br label %[[LOOP_NEXT:.*]]
// LLVM: [[LOOP_END]]:
// LLVM:   ret void

// OGCG: %[[ARR:.*]] = alloca [2 x i32], align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[ARR]], ptr align 4 @[[FUN2_ARR]], i64 8, i1 false)

void func3() {
  int arr[2] = {5, 6};

  int idx = 1;
  int e = arr[idx];
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
// CIR: %[[IDX:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["idx", init]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[V0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store{{.*}} %[[V0]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_1_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[OFFSET_0]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[V1:.*]] = cir.const #cir.int<6> : !s32i
// CIR: cir.store{{.*}} %[[V1]], %[[ELE_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX_V:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[IDX_V]], %[[IDX]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_IDX:.*]] = cir.load{{.*}} %[[IDX]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[TMP_IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: %[[ELE_TMP:.*]] = cir.load{{.*}} %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}} %[[ELE_TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z5func3v(){{.*}}
// LLVM:  %[[ARR:.*]] = alloca [2 x i32], i64 1, align 4
// LLVM:  %[[IDX:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:  store i32 5, ptr %[[ARR_PTR]], align 4
// LLVM:  %[[ELE_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// LLVM:  store i32 6, ptr %[[ELE_1_PTR]], align 4
// LLVM:  store i32 1, ptr %[[IDX]], align 4
// LLVM:  %[[TMP1:.*]] = load i32, ptr %[[IDX]], align 4
// LLVM:  %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:  %[[IDX_I64:.*]] = sext i32 %[[TMP1]] to i64
// LLVM:  %[[ELE:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 %[[IDX_I64]]
// LLVM:  %[[TMP2:.*]] = load i32, ptr %[[ELE]], align 4
// LLVM:  store i32 %[[TMP2]], ptr %[[INIT]], align 4

// OGCG:  %[[ARR:.*]] = alloca [2 x i32], align 4
// OGCG:  %[[IDX:.*]] = alloca i32, align 4
// OGCG:  %[[INIT:.*]] = alloca i32, align 4
// OGCG:  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[ARR]], ptr align 4 @[[FUN3_ARR]], i64 8, i1 false)
// OGCG:  store i32 1, ptr %[[IDX]], align 4
// OGCG:  %[[TMP:.*]] = load i32, ptr %[[IDX]], align 4
// OGCG:  %[[IDX_I64:.*]] = sext i32 %[[TMP]] to i64
// OGCG:  %[[ELE:.*]] = getelementptr inbounds [2 x i32], ptr %[[ARR]], i64 0, i64 %[[IDX_I64]]
// OGCG:  %[[TMP_2:.*]] = load i32, ptr %[[ELE]], align 4
// OGCG:  store i32 %[[TMP_2:.*]], ptr %[[INIT]], align 4

void func4() {
  int arr[2][1] = {{5}, {6}};
  int e = arr[1][0];
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 1> x 2>, !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>, ["arr", init]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>> -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_0_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store{{.*}} %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_1:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[OFFSET]] : (!cir.ptr<!cir.array<!s32i x 1>>, !s64i) -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: %[[V_1_0:.*]] = cir.const #cir.int<6> : !s32i
// CIR: cir.store{{.*}} %[[V_1_0]], %[[ARR_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[IDX_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>> -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[IDX_1]] : (!cir.ptr<!cir.array<!s32i x 1>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: %[[ELE_0:.*]] = cir.ptr_stride %[[ARR_1_PTR]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[ELE_0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}} %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z5func4v(){{.*}}
// LLVM:  %[[ARR:.*]] = alloca [2 x [1 x i32]], i64 1, align 4
// LLVM:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR]], i32 0
// LLVM:  %[[ARR_0_0:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i32 0
// LLVM:  store i32 5, ptr %[[ARR_0_0]], align 4
// LLVM:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// LLVM:  %[[ARR_1_0:.*]] = getelementptr i32, ptr %[[ARR_1]], i32 0
// LLVM:  store i32 6, ptr %[[ARR_1_0]], align 4
// LLVM:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR]], i32 0
// LLVM:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// LLVM:  %[[ARR_1_0:.*]] = getelementptr i32, ptr %[[ARR_1]], i32 0
// LLVM:  %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_1_0]], i64 0
// LLVM:  %[[TMP:.*]] = load i32, ptr %[[ELE_PTR]], align 4
// LLVM:  store i32 %[[TMP]], ptr %[[INIT]], align 4

// OGCG: %[[ARR:.*]] = alloca [2 x [1 x i32]], align 4
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[ARR]], ptr align 4 @[[FUN4_ARR]], i64 8, i1 false)
// OGCG: %[[ARR_1:.*]] = getelementptr inbounds [2 x [1 x i32]], ptr %[[ARR]], i64 0, i64 1
// OGCG: %[[ARR_1_0:.*]] = getelementptr inbounds [1 x i32], ptr %[[ARR_1]], i64 0, i64 0
// OGCG: %[[TMP:.*]] = load i32, ptr %[[ARR_1_0]], align 4
// OGCG: store i32 %[[TMP]], ptr %[[INIT]], align 4

void func5() {
  int arr[2][1] = {{5}};
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 1> x 2>, !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>, ["arr", init]
// CIR: %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, ["arrayinit.temp", init]
// CIR: %[[ARR_0:.*]] = cir.cast array_to_ptrdecay %0 : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>> -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_0_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_0]] : !cir.ptr<!cir.array<!s32i x 1>> -> !cir.ptr<!s32i>
// CIR: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store{{.*}} %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_1:.*]] = cir.ptr_stride %[[ARR_0]], %[[OFFSET]] : (!cir.ptr<!cir.array<!s32i x 1>>, !s64i) -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: cir.store{{.*}} %[[ARR_1]], %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
// CIR: %[[TWO:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[ARR_END:.*]] = cir.ptr_stride %[[ARR_0]], %[[TWO]] : (!cir.ptr<!cir.array<!s32i x 1>>, !s64i) -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR: cir.do {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, !cir.ptr<!cir.array<!s32i x 1>>
// CIR:   %[[ZERO:.*]] = cir.const #cir.zero : !cir.array<!s32i x 1>
// CIR:   cir.store{{.*}} %[[ZERO]], %[[ARR_CUR]] : !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[ONE]] : (!cir.ptr<!cir.array<!s32i x 1>>, !s64i) -> !cir.ptr<!cir.array<!s32i x 1>>
// CIR:   cir.store{{.*}} %[[ARR_NEXT]], %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, !cir.ptr<!cir.array<!s32i x 1>>
// CIR:   %[[CMP:.*]] = cir.cmp(ne, %[[ARR_CUR]], %[[ARR_END]]) : !cir.ptr<!cir.array<!s32i x 1>>, !cir.bool
// CIR:   cir.condition(%[[CMP]])
// CIR: }

// LLVM: define{{.*}} void @_Z5func5v(){{.*}}
// LLVM:   %[[ARR:.*]] = alloca [2 x [1 x i32]], i64 1, align 4
// LLVM:   %[[TMP:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR]], i32 0
// LLVM:   %[[ARR_0:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i32 0
// LLVM:   store i32 5, ptr %[[ARR_0]], align 4
// LLVM:   %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// LLVM:   store ptr %[[ARR_1]], ptr %[[TMP]], align 8
// LLVM:   %[[END_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 2
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_NEXT:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[END_PTR]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   store [1 x i32] zeroinitializer, ptr %[[CUR]], align 4
// LLVM:   %[[NEXT:.*]] = getelementptr [1 x i32], ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[TMP]], align 8
// LLVM:   br label %[[LOOP_NEXT:.*]]
// LLVM: [[LOOP_END]]:
// LLVM:   ret void

// ORGC: %[[ARR:.*]] = alloca [2 x [1 x i32]], align 4
// ORGC: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[ARR]], ptr align 4 @[[FUN5_ARR]], i64 8, i1 false)

void func6() {
  int x = 4;
  int arr[2] = { x, 5 };
}

// CIR: %[[VAR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
// CIR: %[[V:.*]] = cir.const #cir.int<4> : !s32i
// CIR: cir.store{{.*}} %[[V]], %[[VAR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>> -> !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[VAR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}} %[[TMP]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[OFFSET]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[V1:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store{{.*}} %[[V1]], %[[ELE_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z5func6v(){{.*}}
// LLVM:  %[[VAR:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[ARR:.*]] = alloca [2 x i32], i64 1, align 4
// LLVM:  store i32 4, ptr %[[VAR]], align 4
// LLVM:  %[[ELE_0:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:  %[[TMP:.*]] = load i32, ptr %[[VAR]], align 4
// LLVM:  store i32 %[[TMP]], ptr %[[ELE_0]], align 4
// LLVM:  %[[ELE_1:.*]] = getelementptr i32, ptr %[[ELE_0]], i64 1
// LLVM:  store i32 5, ptr %[[ELE_1]], align 4

// OGCG:  %[[VAR:.*]] = alloca i32, align 4
// OGCG:  %[[ARR:.*]] = alloca [2 x i32], align 4
// OGCG:  store i32 4, ptr %[[VAR]], align 4
// OGCG:  %[[ELE_0:.*]] = load i32, ptr %[[VAR]], align 4
// OGCG:  store i32 %[[ELE_0]], ptr %[[ARR]], align 4
// OGCG:  %[[ELE_1:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 1
// OGCG:  store i32 5, ptr %[[ELE_1:.*]], align 4

void func7() {
  int* arr[1] = {};
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!cir.ptr<!s32i> x 1>, !cir.ptr<!cir.array<!cir.ptr<!s32i> x 1>>, ["arr", init]
// CIR: %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["arrayinit.temp", init]
// CIR: %[[ARR_0:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 1>> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.store{{.*}} %[[ARR_0]], %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_END:.*]] = cir.ptr_stride %[[ARR_0]], %[[ONE]] : (!cir.ptr<!cir.ptr<!s32i>>, !s64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.do {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[NULL_PTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %[[NULL_PTR]], %[[ARR_CUR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[ARR_NEXT:.*]] = cir.ptr_stride %[[ARR_CUR]], %[[ONE]] : (!cir.ptr<!cir.ptr<!s32i>>, !s64i) -> !cir.ptr<!cir.ptr<!s32i>>
// CIR:   cir.store{{.*}} %[[ARR_NEXT]], %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR:   cir.yield
// CIR: } while {
// CIR:   %[[ARR_CUR:.*]] = cir.load{{.*}} %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:   %[[CMP:.*]] = cir.cmp(ne, %[[ARR_CUR]], %[[ARR_END]]) : !cir.ptr<!cir.ptr<!s32i>>, !cir.bool
// CIR:   cir.condition(%[[CMP]])
// CIR: }

// LLVM: define{{.*}} void @_Z5func7v(){{.*}}
// LLVM:   %[[ARR:.*]] = alloca [1 x ptr], i64 1, align 8
// LLVM:   %[[TMP:.*]] = alloca ptr, i64 1, align 8
// LLVM:   %[[ARR_PTR:.*]] = getelementptr ptr, ptr %[[ARR]], i32 0
// LLVM:   store ptr %[[ARR_PTR]], ptr %[[TMP]], align 8
// LLVM:   %[[END_PTR:.*]] = getelementptr ptr, ptr %[[ARR_PTR]], i64 1
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_NEXT:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[END_PTR]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:   store ptr null, ptr %[[CUR]], align 8
// LLVM:   %[[NEXT:.*]] = getelementptr ptr, ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[TMP]], align 8
// LLVM:   br label %[[LOOP_NEXT:.*]]
// LLVM: [[LOOP_END]]:
// LLVM:   ret void

// OGCG: %[[ARR:.*]] = alloca [1 x ptr], align 8
// OGCG: call void @llvm.memset.p0.i64(ptr align 8 %[[ARR]], i8 0, i64 8, i1 false)

void func8(int arr[10]) {
  int e = arr[0];
  int e2 = arr[1];
}

// CIR: cir.func{{.*}} @_Z5func8Pi(%[[ARG:.*]]: !cir.ptr<!s32i>
// CIR:  %[[ARR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arr", init]
// CIR:  %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR:  %[[INIT_2:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e2", init]
// CIR:  cir.store{{.*}} %[[ARG]], %[[ARR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR:  %[[IDX:.*]] = cir.const #cir.int<0> : !s32i
// CIR:  %[[TMP_1:.*]] = cir.load{{.*}} %[[ARR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:  %[[ELE_0:.*]] = cir.ptr_stride %[[TMP_1]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR:  %[[TMP_2:.*]] = cir.load{{.*}} %[[ELE_0]] : !cir.ptr<!s32i>, !s32i
// CIR:  cir.store{{.*}} %[[TMP_2]], %[[INIT]] : !s32i, !cir.ptr<!s32i>
// CIR:  %[[IDX_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:  %[[TMP_3:.*]] = cir.load{{.*}} %[[ARR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR:  %[[ELE_1:.*]] = cir.ptr_stride %[[TMP_3]], %[[IDX_1]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR:  %[[TMP_4:.*]] = cir.load{{.*}} %[[ELE_1]] : !cir.ptr<!s32i>, !s32i
// CIR:  cir.store{{.*}} %[[TMP_4]], %[[INIT_2]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z5func8Pi(ptr %[[ARG:.*]]){{.*}}
// LLVM:  %[[ARR:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM:  %[[INIT_2:.*]] = alloca i32, i64 1, align 4
// LLVM:  store ptr %[[ARG]], ptr %[[ARR]], align 8
// LLVM:  %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// LLVM:  %[[ELE_0:.*]] = getelementptr i32, ptr %[[TMP_1]], i64 0
// LLVM:  %[[TMP_2:.*]] = load i32, ptr %[[ELE_0]], align 4
// LLVM:  store i32 %[[TMP_2]], ptr %[[INIT]], align 4
// LLVM:  %[[TMP_3:.*]] = load ptr, ptr %[[ARR]], align 8
// LLVM:  %[[ELE_1:.*]] = getelementptr i32, ptr %[[TMP_3]], i64 1
// LLVM:  %[[TMP_4:.*]] = load i32, ptr %[[ELE_1]], align 4
// LLVM:  store i32 %[[TMP_4]], ptr %[[INIT_2]], align 4

// OGCG: %[[ARR:.*]] = alloca ptr, align 8
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: %[[INIT_2:.*]] = alloca i32, align 4
// OGCG: store ptr {{%.*}}, ptr %[[ARR]], align 8
// OGCG: %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// OGCG: %[[ELE_0:.*]] = getelementptr inbounds i32, ptr %[[TMP_1]], i64 0
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[ELE_0]], align 4
// OGCG: store i32 %[[TMP_2]], ptr %[[INIT]], align 4
// OGCG: %[[TMP_3:.*]] = load ptr, ptr %[[ARR]], align 8
// OGCG: %[[ELE_1:.*]] = getelementptr inbounds i32, ptr %[[TMP_3]], i64 1
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[ELE_1]], align 4
// OGCG: store i32 %[[TMP_2]], ptr %[[INIT_2]], align 4

void func9(int arr[10][5]) {
  int e = arr[1][2];
}

// CIR: cir.func{{.*}} @_Z5func9PA5_i(%[[ARG:.*]]: !cir.ptr<!cir.array<!s32i x 5>>
// CIR:  %[[ARR:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>, ["arr", init]
// CIR:  %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR:  cir.store{{.*}} %[[ARG]], %[[ARR]] : !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>
// CIR:  %[[IDX:.*]] = cir.const #cir.int<2> : !s32i
// CIR:  %[[IDX_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR:  %[[TMP_1:.*]] = cir.load{{.*}} %[[ARR]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>, !cir.ptr<!cir.array<!s32i x 5>>
// CIR:  %[[ARR_1:.*]] = cir.ptr_stride %[[TMP_1]], %[[IDX_1]] : (!cir.ptr<!cir.array<!s32i x 5>>, !s32i) -> !cir.ptr<!cir.array<!s32i x 5>>
// CIR:  %[[ARR_1_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 5>> -> !cir.ptr<!s32i>
// CIR:  %[[ARR_1_2:.*]] = cir.ptr_stride %[[ARR_1_PTR]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR:  %[[TMP_2:.*]] = cir.load{{.*}} %[[ARR_1_2]] : !cir.ptr<!s32i>, !s32i
// CIR:  cir.store{{.*}} %[[TMP_2]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z5func9PA5_i(ptr %[[ARG:.*]]){{.*}}
// LLVM:  %[[ARR:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM:  store ptr %[[ARG]], ptr %[[ARR]], align 8
// LLVM:  %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// LLVM:  %[[ARR_1:.*]] = getelementptr [5 x i32], ptr %[[TMP_1]], i64 1
// LLVM:  %[[ARR_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_1]], i32 0
// LLVM:  %[[ARR_1_2:.*]] = getelementptr i32, ptr %[[ARR_1_PTR]], i64 2
// LLVM:  %[[TMP_2:.*]] = load i32, ptr %[[ARR_1_2]], align 4
// LLVM:  store i32 %[[TMP_2]], ptr %[[INIT]], align 4

// OGCG: %[[ARR:.*]] = alloca ptr, align 8
// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: store ptr {{%.*}}, ptr %[[ARR]], align 8
// OGCG: %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// OGCG: %[[ARR_1:.*]] = getelementptr inbounds [5 x i32], ptr %[[TMP_1]], i64 1
// OGCG: %[[ARR_1_2:.*]] = getelementptr inbounds [5 x i32], ptr %[[ARR_1]], i64 0, i64 2
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[ARR_1_2]], align 4
// OGCG: store i32 %[[TMP_2]], ptr %[[INIT]], align 4

void func10(int *a) {
  int e = a[5];
}

// CIR: cir.func{{.*}} @_Z6func10Pi(%[[ARG:.*]]: !cir.ptr<!s32i>
// CIR: %[[ARR:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["a", init]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: cir.store{{.*}} %[[ARG]], %[[ARR]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[IDX:.*]] = cir.const #cir.int<5> : !s32i
// CIR: %[[TMP_1:.*]] = cir.load{{.*}} %[[ARR]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: %[[ELE:.*]] = cir.ptr_stride %[[TMP_1]], %[[IDX]] : (!cir.ptr<!s32i>, !s32i) -> !cir.ptr<!s32i>
// CIR: %[[TMP_2:.*]] = cir.load{{.*}} %[[ELE]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}} %[[TMP_2]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define{{.*}} void @_Z6func10Pi(ptr %[[ARG:.*]]){{.*}} {
// LLVM:  %[[ARR:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM:  store ptr %[[ARG]], ptr %[[ARR]], align 8
// LLVM:  %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// LLVM:  %[[ELE:.*]] = getelementptr i32, ptr %[[TMP_1]], i64 5
// LLVM:  %[[TMP_2:.*]] = load i32, ptr %[[ELE]], align 4
// LLVM:  store i32 %[[TMP_2]], ptr %[[INIT]], align 4

// OGCG:  %[[ARR:.*]] = alloca ptr, align 8
// OGCG:  %[[INIT:.*]] = alloca i32, align 4
// OGCG:  store ptr {{%.*}}, ptr %[[ARR]], align 8
// OGCG:  %[[TMP_1:.*]] = load ptr, ptr %[[ARR]], align 8
// OGCG:  %[[ELE:.*]] = getelementptr inbounds i32, ptr %[[TMP_1]], i64 5
// OGCG:  %[[TMP_2:.*]] = load i32, ptr %[[ELE]], align 4
// OGCG:  store i32 %[[TMP_2]], ptr %[[INIT]], align 4

void func11() { int _Complex a[4]; }

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!cir.complex<!s32i> x 4>, !cir.ptr<!cir.array<!cir.complex<!s32i> x 4>>, ["a"]

// LLVM: %[[ARR:.*]] = alloca [4 x { i32, i32 }], i64 1, align 16

// OGCG: %[[ARR:.*]] = alloca [4 x { i32, i32 }], align 16

void func12() {
  struct Point {
    int x;
    int y;
  };

  Point a[4];
}

// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!rec_Point x 4>, !cir.ptr<!cir.array<!rec_Point x 4>>, ["a"]

// LLVM: %[[ARR:.*]] = alloca [4 x %struct.Point], i64 1, align 16

// OGCG: %[[ARR:.*]] = alloca [4 x %struct.Point], align 16

void array_with_complex_elements() {
  _Complex float arr[2] = {{1.1f, 2.2f}, {3.3f, 4.4f}};
}

// CIR: %[[ARR_ADDR:.*]] = cir.alloca !cir.array<!cir.complex<!cir.float> x 2>, !cir.ptr<!cir.array<!cir.complex<!cir.float> x 2>>, ["arr", init]
// CIR: %[[ARR_0:.*]] = cir.cast array_to_ptrdecay %[[ARR_ADDR]] : !cir.ptr<!cir.array<!cir.complex<!cir.float> x 2>> -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[CONST_COMPLEX_0:.*]] = cir.const #cir.const_complex<#cir.fp<1.100000e+00> : !cir.float, #cir.fp<2.200000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[CONST_COMPLEX_0]], %[[ARR_0]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[IDX_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_1:.*]] = cir.ptr_stride %1, %[[IDX_1]] : (!cir.ptr<!cir.complex<!cir.float>>, !s64i) -> !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[CONST_COMPLEX_1:.*]] = cir.const #cir.const_complex<#cir.fp<3.300000e+00> : !cir.float, #cir.fp<4.400000e+00> : !cir.float> : !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[CONST_COMPLEX_1]], %[[ARR_1]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[ARR_ADDR:.*]] = alloca [2 x { float, float }], i64 1, align 16
// LLVM: %[[ARR_0:.*]] = getelementptr { float, float }, ptr %[[ARR_ADDR]], i32 0
// LLVM: store { float, float } { float 0x3FF19999A0000000, float 0x40019999A0000000 }, ptr %[[ARR_0]], align 8
// LLVM: %[[ARR_1:.*]] = getelementptr { float, float }, ptr %[[ARR_0]], i64 1
// LLVM: store { float, float } { float 0x400A666660000000, float 0x40119999A0000000 }, ptr %[[ARR_1]], align 8

// OGCG: %[[ARR_ADDR:.*]] = alloca [2 x { float, float }], align 16
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[ARR_ADDR]], ptr align 16 @__const._Z27array_with_complex_elementsv.arr, i64 16, i1 false)
