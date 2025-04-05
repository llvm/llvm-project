// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

int a[10];
// CHECK: cir.global external @a = #cir.zero : !cir.array<!s32i x 10>

int aa[10][5];
// CHECK: cir.global external @aa = #cir.zero : !cir.array<!cir.array<!s32i x 5> x 10>

extern int b[10];
// CHECK: cir.global external @b = #cir.zero : !cir.array<!s32i x 10>

extern int bb[10][5];
// CHECK: cir.global external @bb = #cir.zero : !cir.array<!cir.array<!s32i x 5> x 10>

int c[10] = {};
// CHECK: cir.global external @c = #cir.zero : !cir.array<!s32i x 10>

int d[3] = {1, 2, 3};
// CHECK: cir.global external @d = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<3> : !s32i]> : !cir.array<!s32i x 3>

int dd[3][2] = {{1, 2}, {3, 4}, {5, 6}};
// CHECK: cir.global external @dd = #cir.const_array<[#cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<3> : !s32i, #cir.int<4> : !s32i]> : !cir.array<!s32i x 2>, #cir.const_array<[#cir.int<5> : !s32i, #cir.int<6> : !s32i]> : !cir.array<!s32i x 2>]> : !cir.array<!cir.array<!s32i x 2> x 3>

int e[10] = {1, 2};
// CHECK: cir.global external @e = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i], trailing_zeros> : !cir.array<!s32i x 10>

int f[5] = {1, 2};
// CHECK: cir.global external @f = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i, #cir.int<0> : !s32i]> : !cir.array<!s32i x 5>

void func() {
  int arr[10];

  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 10>, !cir.ptr<!cir.array<!s32i x 10>>, ["arr"]
}

void func2() {
  int arr[2] = {5};

  // CHECK: %[[ARR2:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
  // CHECK: %[[ELE_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init]
  // CHECK: %[[ARR_2_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR2]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
  // CHECK: %[[V1:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store %[[V1]], %[[ARR_2_PTR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_2_PTR]] : !cir.ptr<!s32i>, %[[OFFSET_0]] : !s64i), !cir.ptr<!s32i>
  // CHECK: cir.store %[[ELE_PTR]], %[[ELE_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[LOAD_1:.*]] = cir.load %[[ELE_ALLOCA]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[V2:.*]] = cir.const #cir.int<0> : !s32i
  // CHECK: cir.store %[[V2]], %[[LOAD_1]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET_1:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ELE_1_PTR:.*]] = cir.ptr_stride(%[[LOAD_1]] : !cir.ptr<!s32i>, %[[OFFSET_1]] : !s64i), !cir.ptr<!s32i>
  // CHECK: cir.store %[[ELE_1_PTR]], %[[ELE_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
}

void func3() {
  int arr[2] = {5, 6};

  // CHECK: %[[ARR3:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
  // CHECK: %[[ARR_3_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR3]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
  // CHECK: %[[V0:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store %[[V0]], %[[ARR_3_PTR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET_0:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ELE_1_PTR:.*]] = cir.ptr_stride(%[[ARR_3_PTR]] : !cir.ptr<!s32i>, %[[OFFSET_0]] : !s64i), !cir.ptr<!s32i>
  // CHECK: %[[V1:.*]] = cir.const #cir.int<6> : !s32i
  // CHECK: cir.store %[[V1]], %[[ELE_1_PTR]] : !s32i, !cir.ptr<!s32i>
}

void func4() {
  int arr[2][1] = {{5}, {6}};

  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 1> x 2>, !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>, ["arr", init]
  // CHECK: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>), !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: %[[ARR_0_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
  // CHECK: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ARR_1:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: %[[ARR_1_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_1]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
  // CHECK: %[[V_1_0:.*]] = cir.const #cir.int<6> : !s32i
  // CHECK: cir.store %[[V_1_0]], %[[ARR_1_PTR]] : !s32i, !cir.ptr<!s32i>
}

void func5() {
  int arr[2][1] = {{5}};

  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.array<!s32i x 1> x 2>, !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>, ["arr", init]
  // CHECK: %[[ARR_PTR:.*]] = cir.alloca !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, ["arrayinit.temp", init]
  // CHECK: %[[ARR_0:.*]] = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<!cir.array<!s32i x 1> x 2>>), !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: %[[ARR_0_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR_0]] : !cir.ptr<!cir.array<!s32i x 1>>), !cir.ptr<!s32i>
  // CHECK: %[[V_0_0:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store %[[V_0_0]], %[[ARR_0_PTR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %6 = cir.ptr_stride(%[[ARR_0]] : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: cir.store %6, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
  // CHECK: %7 = cir.load %[[ARR_PTR]] : !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>, !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: %8 = cir.const #cir.zero : !cir.array<!s32i x 1>
  // CHECK: cir.store %8, %7 : !cir.array<!s32i x 1>, !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: %[[OFFSET_1:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %10 = cir.ptr_stride(%7 : !cir.ptr<!cir.array<!s32i x 1>>, %[[OFFSET_1]] : !s64i), !cir.ptr<!cir.array<!s32i x 1>>
  // CHECK: cir.store %10, %[[ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 1>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 1>>>
}

void func6() {
  int x = 4;
  int arr[2] = { x, 5 };

  // CHECK: %[[VAR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!s32i x 2>, !cir.ptr<!cir.array<!s32i x 2>>, ["arr", init]
  // CHECK: %[[V:.*]] = cir.const #cir.int<4> : !s32i
  // CHECK: cir.store %[[V]], %[[VAR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
  // CHECK: %[[TMP:.*]] = cir.load %[[VAR]] : !cir.ptr<!s32i>, !s32i
  // CHECK: cir.store %[[TMP]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
  // CHECK: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[ARR_PTR]] : !cir.ptr<!s32i>, %[[OFFSET]] : !s64i), !cir.ptr<!s32i>
  // CHECK: %[[V1:.*]] = cir.const #cir.int<5> : !s32i
  // CHECK: cir.store %[[V1]], %[[ELE_PTR]] : !s32i, !cir.ptr<!s32i>
}

void func7() {
  int* arr[1] = {};

  // CHECK: %[[ARR:.*]] = cir.alloca !cir.array<!cir.ptr<!s32i> x 1>, !cir.ptr<!cir.array<!cir.ptr<!s32i> x 1>>, ["arr", init]
  // CHECK: %[[ARR_TMP:.*]] = cir.alloca !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, ["arrayinit.temp", init]
  // CHECK: %[[ARR_PTR:.*]] = cir.cast(array_to_ptrdecay, %[[ARR]] : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 1>>), !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: cir.store %[[ARR_PTR]], %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
  // CHECK: %[[TMP:.*]] = cir.load %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[NULL_PTR:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
  // CHECK: cir.store %[[NULL_PTR]], %[[TMP]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: %[[OFFSET:.*]] = cir.const #cir.int<1> : !s64i
  // CHECK: %[[ELE_PTR:.*]] = cir.ptr_stride(%[[TMP]] : !cir.ptr<!cir.ptr<!s32i>>, %[[OFFSET]] : !s64i), !cir.ptr<!cir.ptr<!s32i>>
  // CHECK: cir.store %[[ELE_PTR]], %[[ARR_TMP]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s32i>>>
}

void func8(int p[10]) {}
// CHECK: cir.func @func8(%arg0: !cir.ptr<!s32i>
// CHECK: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["p", init]

void func9(int pp[10][5]) {}
// CHECK: cir.func @func9(%arg0: !cir.ptr<!cir.array<!s32i x 5>>
// CHECK: cir.alloca !cir.ptr<!cir.array<!s32i x 5>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 5>>>
