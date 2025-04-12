// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int a[10];
// CIR: cir.global external @a = #cir.zero : !cir.array<!s32i x 10>

// LLVM: @a = dso_local global [10 x i32] zeroinitializer

// OGCG: @a = global [10 x i32] zeroinitializer

int aa[10][5];
// CIR: cir.global external @aa = #cir.zero : !cir.array<!cir.array<!s32i x 5> x 10>

// LLVM: @aa = dso_local global [10 x [5 x i32]] zeroinitializer

// OGCG: @aa = global [10 x [5 x i32]] zeroinitializer

extern int b[10];
// CIR: cir.global external @b = #cir.zero : !cir.array<!s32i x 10>

// LLVM: @b = dso_local global [10 x i32] zeroinitializer

extern int bb[10][5];
// CIR: cir.global external @bb = #cir.zero : !cir.array<!cir.array<!s32i x 5> x 10>

// LLVM: @bb = dso_local global [10 x [5 x i32]] zeroinitializer

int c[10] = {};
// CIR: cir.global external @c = #cir.zero : !cir.array<!s32i x 10>

// LLVM: @c = dso_local global [10 x i32] zeroinitializer

// OGCG: @c = global [10 x i32] zeroinitializer

int d[3] = {1, 2, 3};
// CIR: cir.global external @d = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>

// LLVM: @d = dso_local global [3 x i32] [i32 1, i32 2, i32 3]

// OGCG: @d = global [3 x i32] [i32 1, i32 2, i32 3]

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CIR: cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 3>

// LLVM: @dd = dso_local global [3 x [2 x i32]] [
// LLVM: [2 x i32] [i32 1, i32 2], [2 x i32]
// LLVM: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

// OGCG: @dd = global [3 x [2 x i32]] [
// OGCG: [2 x i32] [i32 1, i32 2], [2 x i32]
// OGCG: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

int e[10] = {1, 2};
// CIR: cir.global external @e = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>

// LLVM: @e = dso_local global [10 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0]

int f[5] = {1, 2};
// CIR: cir.global external @f = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 5>

// LLVM: @f = dso_local global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

// OGCG: @f = global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

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
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>>), !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[IDX]] : !s32i), !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR" cir.store %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 10>>), !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[IDX]] : !s32i), !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR" cir.store %[[TMP]], %[[INIT_2]] : !s32i, !cir.ptr<!s32i>

// LLVM: define void @func()
// LLVM-NEXT: %[[ARR:.*]] = alloca [10 x i32], i64 1, align 16
// LLVM-NEXT: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT: %[[INIT_2:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT: %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM-NEXT: %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 0
// LLVM-NEXT: %[[TMP_1:.*]] = load i32, ptr %[[ELE_PTR]], align 4
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
// CIR: %[[ELE_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init]
// CIR: %[[ARR_2_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR2]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CIR: %[[V1:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[V1]], %[[ARR_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_2_PTR]] : !cir.ptr<!s32i>, %[[OFFSET_0]] : !s64i), !cir.ptr<!s32i>
// CIR: cir.store %[[ELE_PTR]], %[[ELE_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[LOAD_1:.*]] = cir.load %[[ELE_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CIR: %[[V2:.*]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store %[[V2]], %[[LOAD_1]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_1_PTR:.*]] = cir.ptr_stride(%[[LOAD_1]] : !cir.ptr<!s32i>, %[[OFFSET_1]] : !s64i), !cir.ptr<!s32i>
// CIR: cir.store %[[ELE_1_PTR]], %[[ELE_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// LLVM: define void @func2()
// LLVM:  %[[ARR:.*]] = alloca [2 x i32], i64 1, align 4
// LLVM:  %[[TMP:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:  store i32 5, ptr %[[ARR_PTR]], align 4
// LLVM:  %[[ELE_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// LLVM:  store ptr %[[ELE_1_PTR]], ptr %[[TMP]], align 8
// LLVM:  %[[TMP2:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:  store i32 0, ptr %[[TMP2]], align 4
// LLVM:  %[[ELE_1:.*]] = getelementptr i32, ptr %[[TMP2]], i64 1
// LLVM:  store ptr %[[ELE_1]], ptr %[[TMP]], align 8

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
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CIR: %[[V0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[V0]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_1_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[OFFSET_0]] : !s64i), !cir.ptr<!s32i>
// CIR: %[[V1:.*]] = cir.const #cir.int<6> : !s32i
// CIR: cir.store %[[V1]], %[[ELE_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX_V:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store %[[IDX_V]], %[[IDX]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_IDX:.*]] = cir.load %[[IDX]] : !cir.ptr<!s32i>, !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[TMP_IDX]] : !s32i), !cir.ptr<!s32i>
// CIR: %[[ELE_TMP:.*]] = cir.load %[[ELE_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[ELE_TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define void @func3()
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
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_0_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
// CIR: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_1:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
// CIR: %[[V_1_0:.*]] = cir.const #cir.int<6> : !s32i
// CIR: cir.store %[[V_1_0]], %[[ARR_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[IDX:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[IDX_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, %[[IDX_1]] : !s32i), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_1_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
// CIR: %[[ELE_0:.*]] = cir.ptr_stride(%[[ARR_1_PTR]] : !cir.ptr<!s32i>, %[[IDX]] : !s32i), !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[ELE_0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>

// LLVM: define void @func4()
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
// CIR: %[[ARR_0:.*]] = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[ARR_0_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_0]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
// CIR: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %6 = cir.ptr_stride(%[[ARR_0]] : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: cir.store %6, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
// CIR: %7 = cir.load %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %8 = cir.const #cir.zero : !cir.array<!s32i x 1>
// CIR: cir.store %8, %7 : !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>
// CIR: %[[OFFSET_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %10 = cir.ptr_stride(%7 : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET_1]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
// CIR: cir.store %10, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>

// LLVM: define void @func5()
// LLVM:  %[[ARR:.*]] = alloca [2 x [1 x i32]], i64 1, align 4
// LLVM:  %[[TMP:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR]], i32 0
// LLVM:  %[[ARR_0:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i32 0
// LLVM:  store i32 5, ptr %[[ARR_0]], align 4
// LLVM:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// LLVM:  store ptr %[[ARR_1]], ptr %[[TMP]], align 8
// LLVM:  %[[ARR_1_VAL:.*]] = load ptr, ptr %[[TMP]], align 8
// LLVM:  store [1 x i32] zeroinitializer, ptr %[[ARR_1_VAL]], align 4
// LLVM:  %[[ARR_1_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_1_VAL]], i64 1
// LLVM:  store ptr %[[ARR_1_PTR]], ptr %[[TMP]], align 8

// ORGC: %[[ARR:.*]] = alloca [2 x [1 x i32]], align 4
// ORGC: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[ARR]], ptr align 4 @[[FUN5_ARR]], i64 8, i1 false)

void func6() {
  int x = 4;
  int arr[2] = { x, 5 };
}

// CIR: %[[VAR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
// CIR: %[[V:.*]] = cir.const #cir.int<4> : !s32i
// CIR: cir.store %[[V]], %[[VAR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[VAR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[TMP]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[OFFSET]] : !s64i), !cir.ptr<!s32i>
// CIR: %[[V1:.*]] = cir.const #cir.int<5> : !s32i
// CIR: cir.store %[[V1]], %[[ELE_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: define void @func6()
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
// CIR: %[[ARR_TMP:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["arrayinit.temp", init]
// CIR: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 1>>), !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.store %[[ARR_PTR]], %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
// CIR: %[[TMP:.*]] = cir.load %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[NULL_PTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR: cir.store %[[NULL_PTR]], %[[TMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[TMP]] : !cir.ptr<!cir.ptr<!s32i>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.ptr<!s32i>>
// CIR: cir.store %[[ELE_PTR]], %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>

// LLVM: define void @func7()
// LLVM:  %[[ARR:.*]] = alloca [1 x ptr], i64 1, align 8
// LLVM:  %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// LLVM:  %[[ELE_PTR:.*]] = getelementptr ptr, ptr %[[ARR]], i32 0
// LLVM:  store ptr %[[ELE_PTR]], ptr %[[ALLOCA]], align 8
// LLVM:  %[[TMP:.*]] = load ptr, ptr %[[ALLOCA]], align 8
// LLVM:  store ptr null, ptr %[[TMP]], align 8
// LLVM:  %[[ELE:.*]] = getelementptr ptr, ptr %[[TMP]], i64 1
// LLVM:  store ptr %[[ELE]], ptr %[[ALLOCA]], align 8

// OGCG: %[[ARR:.*]] = alloca [1 x ptr], align 8
// OGCG: call void @llvm.memset.p0.i64(ptr align 8 %[[ARR]], i8 0, i64 8, i1 false)

void func8(int p[10]) {}
// CIR: cir.func @func8(%arg0: !cir.ptr<!s32i>
// CIR: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]

// LLVM: define void @func8(ptr {{%.*}})
// LLVM-NEXT: alloca ptr, i64 1, align 8

// OGCG: alloca ptr, align 8

void func9(int pp[10][5]) {}
// CIR: cir.func @func9(%arg0: !cir.ptr<!cir.array<!s32i x 5>>
// CIR: cir.alloca !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>

// LLVM: define void @func9(ptr {{%.*}})
// LLVM-NEXT: alloca ptr, i64 1, align 8

// OGCG: alloca ptr, align 8
