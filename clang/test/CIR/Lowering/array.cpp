// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s

int a[10];
// CHECK: @a = dso_local global [10 x i32] zeroinitializer

int aa[10][5];
// CHECK: @aa = dso_local global [10 x [5 x i32]] zeroinitializer

extern int b[10];
// CHECK: @b = dso_local global [10 x i32] zeroinitializer

extern int bb[10][5];
// CHECK: @bb = dso_local global [10 x [5 x i32]] zeroinitializer

int c[10] = {};
// CHECK: @c = dso_local global [10 x i32] zeroinitializer

int d[3] = {1, 2, 3};
// CHECK: @d = dso_local global [3 x i32] [i32 1, i32 2, i32 3]

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CHECK: @dd = dso_local global [3 x [2 x i32]] [
// CHECK: [2 x i32] [i32 1, i32 2], [2 x i32]
// CHECK: [i32 3, i32 4], [2 x i32] [i32 5, i32 6]]

int e[10] = {1, 2};
// CHECK: @e = dso_local global [10 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0]

int f[5] = {1, 2};
// CHECK: @f = dso_local global [5 x i32] [i32 1, i32 2, i32 0, i32 0, i32 0]

void func() {
  int arr[10];
  int e = arr[0];
  int e2 = arr[1];
}
// CHECK: define void @_Z4funcv()
// CHECK-NEXT: %[[ARR_ALLOCA:.*]] = alloca [10 x i32], i64 1, align 16
// CHECK-NEXT: %[[INIT:.*]] = alloca i32, i64 1, align 4
// CHECK-NEXT: %[[INIT_2:.*]] = alloca i32, i64 1, align 4
// CHECK-NEXT: %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR_ALLOCA]], i32 0
// CHECK-NEXT: %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 0
// CHECK-NEXT: %[[TMP:.*]] = load i32, ptr %[[ELE_PTR]], align 16
// CHECK-NEXT: store i32 %[[TMP]], ptr %[[INIT]], align 4
// CHECK-NEXT: %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR_ALLOCA]], i32 0
// CHECK-NEXT: %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// CHECK-NEXT: %[[TMP:.*]] = load i32, ptr %[[ELE_PTR]], align 4
// CHECK-NEXT: store i32 %[[TMP]], ptr %[[INIT_2]], align 4

void func2() {
  int arr[2] = {5};
}
// CHECK: define void @_Z5func2v()
// CHECK:  %[[ARR_ALLOCA:.*]] = alloca [2 x i32], i64 1, align 4
// CHECK:  %[[TMP:.*]] = alloca ptr, i64 1, align 8
// CHECK:  %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR_ALLOCA]], i32 0
// CHECK:  store i32 5, ptr %[[ARR_PTR]], align 4
// CHECK:  %[[ELE_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// CHECK:  store ptr %[[ELE_1_PTR]], ptr %[[TMP]], align 8
// CHECK:  %[[TMP2:.*]] = load ptr, ptr %[[TMP]], align 8
// CHECK:  store i32 0, ptr %[[TMP2]], align 4
// CHECK:  %[[ELE_1:.*]] = getelementptr i32, ptr %[[TMP2]], i64 1
// CHECK:  store ptr %[[ELE_1]], ptr %[[TMP]], align 8

void func3() {
  int arr3[2] = {5, 6};
}
// CHECK: define void @_Z5func3v()
// CHECK:  %[[ARR_ALLOCA:.*]] = alloca [2 x i32], i64 1, align 4
// CHECK:  %[[ARR_PTR:.*]] = getelementptr i32, ptr %[[ARR_ALLOCA]], i32 0
// CHECK:  store i32 5, ptr %[[ARR_PTR]], align 4
// CHECK:  %[[ELE_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i64 1
// CHECK:  store i32 6, ptr %[[ELE_1_PTR]], align 4

void func4() {
  int arr[2][1] = {{5}, {6}};
  int e = arr[1][0];
}
// CHECK: define void @_Z5func4v()
// CHECK:  %[[ARR_ALLOCA:.*]] = alloca [2 x [1 x i32]], i64 1, align 4
// CHECK:  %[[INIT:.*]] = alloca i32, i64 1, align 4
// CHECK:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_ALLOCA]], i32 0
// CHECK:  %[[ARR_0_0:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i32 0
// CHECK:  store i32 5, ptr %[[ARR_0_0]], align 4
// CHECK:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// CHECK:  %[[ARR_1_0:.*]] = getelementptr i32, ptr %[[ARR_1]], i32 0
// CHECK:  store i32 6, ptr %[[ARR_1_0]], align 4
// CHECK:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_ALLOCA]], i32 0
// CHECK:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// CHECK:  %[[ARR_1_0:.*]] = getelementptr i32, ptr %[[ARR_1]], i32 0
// CHECK:  %[[ELE_PTR:.*]] = getelementptr i32, ptr %[[ARR_1_0]], i64 0
// CHECK:  %[[TMP:.*]] = load i32, ptr %[[ELE_PTR]], align 4
// CHECK:  store i32 %[[TMP]], ptr %[[INIT]], align 4

void func5() {
  int arr[2][1] = {{5}};
}
// CHECK: define void @_Z5func5v()
// CHECK:  %[[ARR_ALLOCA:.*]] = alloca [2 x [1 x i32]], i64 1, align 4
// CHECK:  %[[TMP:.*]] = alloca ptr, i64 1, align 8
// CHECK:  %[[ARR_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_ALLOCA]], i32 0
// CHECK:  %[[ARR_0:.*]] = getelementptr i32, ptr %[[ARR_PTR]], i32 0
// CHECK:  store i32 5, ptr %[[ARR_0]], align 4
// CHECK:  %[[ARR_1:.*]] = getelementptr [1 x i32], ptr %[[ARR_PTR]], i64 1
// CHECK:  store ptr %[[ARR_1]], ptr %[[TMP]], align 8
// CHECK:  %[[ARR_1_VAL:.*]] = load ptr, ptr %[[TMP]], align 8
// CHECK:  store [1 x i32] zeroinitializer, ptr %[[ARR_1_VAL]], align 4
// CHECK:  %[[ARR_1_PTR:.*]] = getelementptr [1 x i32], ptr %[[ARR_1_VAL]], i64 1
// CHECK:  store ptr %[[ARR_1_PTR]], ptr %[[TMP]], align 8

void func6() {
  int x = 4;
  int arr[2] = { x, 5 };
}
// CHECK: define void @_Z5func6v()
// CHECK:  %[[VAR:.*]] = alloca i32, i64 1, align 4
// CHECK:  %[[ARR:.*]] = alloca [2 x i32], i64 1, align 4
// CHECK:  store i32 4, ptr %[[VAR]], align 4
// CHECK:  %[[ELE_0:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// CHECK:  %[[TMP:.*]] = load i32, ptr %[[VAR]], align 4
// CHECK:  store i32 %[[TMP]], ptr %[[ELE_0]], align 4
// CHECK:  %[[ELE_1:.*]] = getelementptr i32, ptr %[[ELE_0]], i64 1
// CHECK:  store i32 5, ptr %[[ELE_1]], align 4

void func7() {
  int* arr[1] = {};
}
// CHECK: define void @_Z5func7v()
// CHECK:  %[[ARR:.*]] = alloca [1 x ptr], i64 1, align 8
// CHECK:  %[[ALLOCA:.*]] = alloca ptr, i64 1, align 8
// CHECK:  %[[ELE_PTR:.*]] = getelementptr ptr, ptr %[[ARR]], i32 0
// CHECK:  store ptr %[[ELE_PTR]], ptr %[[ALLOCA]], align 8
// CHECK:  %[[TMP:.*]] = load ptr, ptr %[[ALLOCA]], align 8
// CHECK:  store ptr null, ptr %[[TMP]], align 8
// CHECK:  %[[ELE:.*]] = getelementptr ptr, ptr %[[TMP]], i64 1
// CHECK:  store ptr %[[ELE]], ptr %[[ALLOCA]], align 8

void func8(int p[10]) {}
// CHECK: define void @_Z5func8Pi(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8

void func9(int pp[10][5]) {}
// CHECK: define void @_Z5func9PA5_i(ptr {{%.*}})
// CHECK-NEXT: alloca ptr, i64 1, align 8
