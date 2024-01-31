// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

void foo() {
  double bar[] = {9,8,7};
}

//      CHECK: %0 = cir.alloca !cir.array<f64 x 3>, cir.ptr <!cir.array<f64 x 3>>, ["bar"] {alignment = 16 : i64}
// CHECK-NEXT: %1 = cir.const(#cir.const_array<[9.000000e+00, 8.000000e+00, 7.000000e+00]> : !cir.array<f64 x 3>) : !cir.array<f64 x 3>
// CHECK-NEXT: cir.store %1, %0 : !cir.array<f64 x 3>, cir.ptr <!cir.array<f64 x 3>>

void bar(int a, int b, int c) {
  int arr[] = {a,b,c};
}

// CHECK: cir.func @bar
// CHECK:      [[ARR:%.*]] = cir.alloca !cir.array<!s32i x 3>, cir.ptr <!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CHECK-NEXT: cir.store %arg0, [[A:%.*]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: cir.store %arg1, [[B:%.*]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: cir.store %arg2, [[C:%.*]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: [[FI_EL:%.*]] = cir.cast(array_to_ptrdecay, [[ARR]] : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
// CHECK-NEXT: [[LOAD_A:%.*]] = cir.load [[A]] : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.store [[LOAD_A]], [[FI_EL]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: [[ONE:%.*]] = cir.const(#cir.int<1> : !s64i) : !s64i
// CHECK-NEXT: [[SE_EL:%.*]] = cir.ptr_stride(%4 : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: [[LOAD_B:%.*]] = cir.load [[B]] : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.store [[LOAD_B]], [[SE_EL]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: [[TH_EL:%.*]] = cir.ptr_stride(%7 : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK-NEXT: [[LOAD_C:%.*]] = cir.load [[C]] : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: cir.store [[LOAD_C]], [[TH_EL]] : !s32i, cir.ptr <!s32i>
