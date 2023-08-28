// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -Wno-return-stack-address -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

void a0() {
  int a[10];
}

// CHECK: cir.func @_Z2a0v()
// CHECK-NEXT:   %0 = cir.alloca !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}

void a1() {
  int a[10];
  a[0] = 1;
}

// CHECK: cir.func @_Z2a1v()
// CHECK-NEXT:  %0 = cir.alloca !cir.array<!s32i x 10>, cir.ptr <!cir.array<!s32i x 10>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:  %1 = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK-NEXT:  %2 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:  %3 = cir.cast(array_to_ptrdecay, %0 : !cir.ptr<!cir.array<!s32i x 10>>), !cir.ptr<!s32i>
// CHECK-NEXT:  %4 = cir.ptr_stride(%3 : !cir.ptr<!s32i>, %2 : !s32i), !cir.ptr<!s32i>
// CHECK-NEXT:  cir.store %1, %4 : !s32i, cir.ptr <!s32i>

int *a2() {
  int a[4];
  return &a[0];
}

// CHECK: cir.func @_Z2a2v() -> !cir.ptr<!s32i>
// CHECK-NEXT:   %0 = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["__retval"] {alignment = 8 : i64}
// CHECK-NEXT:   %1 = cir.alloca !cir.array<!s32i x 4>, cir.ptr <!cir.array<!s32i x 4>>, ["a"] {alignment = 16 : i64}
// CHECK-NEXT:   %2 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:   %3 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s32i x 4>>), !cir.ptr<!s32i>
// CHECK-NEXT:   %4 = cir.ptr_stride(%3 : !cir.ptr<!s32i>, %2 : !s32i), !cir.ptr<!s32i>
// CHECK-NEXT:   cir.store %4, %0 : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK-NEXT:   %5 = cir.load %0 : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK-NEXT:   cir.return %5 : !cir.ptr<!s32i>

void local_stringlit() {
  const char *s = "whatnow";
}

// CHECK: cir.global "private" constant internal @".str" = #cir.const_array<"whatnow\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8> {alignment = 1 : i64}
// CHECK: cir.func @_Z15local_stringlitv()
// CHECK-NEXT:  %0 = cir.alloca !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>, ["s", init] {alignment = 8 : i64}
// CHECK-NEXT:  %1 = cir.get_global @".str" : cir.ptr <!cir.array<!s8i x 8>>
// CHECK-NEXT:  %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s8i x 8>>), !cir.ptr<!s8i>
// CHECK-NEXT:  cir.store %2, %0 : !cir.ptr<!s8i>, cir.ptr <!cir.ptr<!s8i>>

int multidim(int i, int j) {
  int arr[2][2];
  return arr[i][j];
}

// CHECK: %3 = cir.alloca !cir.array<!cir.array<!s32i x 2> x 2>, cir.ptr <!cir.array<!cir.array<!s32i x 2> x 2>>
// Stride first dimension (stride = 2)
// CHECK: %4 = cir.load %{{.+}} : cir.ptr <!s32i>, !s32i
// CHECK: %5 = cir.cast(array_to_ptrdecay, %3 : !cir.ptr<!cir.array<!cir.array<!s32i x 2> x 2>>), !cir.ptr<!cir.array<!s32i x 2>>
// CHECK: %6 = cir.ptr_stride(%5 : !cir.ptr<!cir.array<!s32i x 2>>, %4 : !s32i), !cir.ptr<!cir.array<!s32i x 2>>
// Stride second dimension (stride = 1)
// CHECK: %7 = cir.load %{{.+}} : cir.ptr <!s32i>, !s32i
// CHECK: %8 = cir.cast(array_to_ptrdecay, %6 : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CHECK: %9 = cir.ptr_stride(%8 : !cir.ptr<!s32i>, %7 : !s32i), !cir.ptr<!s32i>

// Should globally zero-initialize null arrays.
int globalNullArr[] = {0, 0};
// CHECK: cir.global external @globalNullArr = #cir.zero : !cir.array<!s32i x 2>

// Should implicitly zero-initialize global array elements.
struct S {
  int i;
} arr[3] = {{1}};
// CHECK: cir.global external @arr = #cir.const_array<[#cir.const_struct<{#cir.int<1> : !s32i}> : !ty_22S22, #cir.zero : !ty_22S22, #cir.zero : !ty_22S22]> : !cir.array<!ty_22S22 x 3>

void testPointerDecaySubscriptAccess(int arr[]) {
// CHECK: cir.func @{{.+}}testPointerDecaySubscriptAccess
  arr[1];
  // CHECK: %[[#BASE:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CHECK: %[[#DIM1:]] = cir.const(#cir.int<1> : !s32i) : !s32i
  // CHECK: cir.ptr_stride(%[[#BASE]] : !cir.ptr<!s32i>, %[[#DIM1]] : !s32i), !cir.ptr<!s32i>
}

void testPointerDecayedArrayMultiDimSubscriptAccess(int arr[][3]) {
// CHECK: cir.func @{{.+}}testPointerDecayedArrayMultiDimSubscriptAccess
  arr[1][2];
  // CHECK: %[[#V1:]] = cir.load %{{.+}} : cir.ptr <!cir.ptr<!cir.array<!s32i x 3>>>, !cir.ptr<!cir.array<!s32i x 3>>
  // CHECK: %[[#V2:]] = cir.const(#cir.int<1> : !s32i) : !s32i
  // CHECK: %[[#V3:]] = cir.ptr_stride(%[[#V1]] : !cir.ptr<!cir.array<!s32i x 3>>, %[[#V2]] : !s32i), !cir.ptr<!cir.array<!s32i x 3>>
  // CHECK: %[[#V4:]] = cir.const(#cir.int<2> : !s32i) : !s32i
  // CHECK: %[[#V5:]] = cir.cast(array_to_ptrdecay, %[[#V3]] : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
  // CHECK: cir.ptr_stride(%[[#V5]] : !cir.ptr<!s32i>, %[[#V4]] : !s32i), !cir.ptr<!s32i>
}
