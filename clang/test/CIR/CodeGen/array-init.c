// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

typedef struct {
  int a;
  long b;
} T;

void buz(int x) {
  T arr[] = { {0, x}, {0, 0} };
}
// CHECK: cir.func @buz
// CHECK-NEXT: [[X_ALLOCA:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK-NEXT: [[ARR:%.*]] = cir.alloca !cir.array<!ty_22T22 x 2>, cir.ptr <!cir.array<!ty_22T22 x 2>>, ["arr", init] {alignment = 16 : i64}
// CHECK-NEXT: cir.store %arg0, [[X_ALLOCA]] : !s32i, cir.ptr <!s32i>
// CHECK-NEXT: [[ARR_INIT:%.*]] = cir.const(#cir.zero : !cir.array<!ty_22T22 x 2>) : !cir.array<!ty_22T22 x 2>
// CHECK-NEXT: cir.store [[ARR_INIT]], [[ARR]] : !cir.array<!ty_22T22 x 2>, cir.ptr <!cir.array<!ty_22T22 x 2>>
// CHECK-NEXT: [[FI_EL:%.*]] = cir.cast(array_to_ptrdecay, [[ARR]] : !cir.ptr<!cir.array<!ty_22T22 x 2>>), !cir.ptr<!ty_22T22>
// CHECK-NEXT: [[A_STORAGE0:%.*]] = cir.get_member [[FI_EL]][0] {name = "a"} : !cir.ptr<!ty_22T22> -> !cir.ptr<!s32i>
// CHECK-NEXT: [[B_STORAGE0:%.*]] = cir.get_member [[FI_EL]][1] {name = "b"} : !cir.ptr<!ty_22T22> -> !cir.ptr<!s64i>
// CHECK-NEXT: [[X_VAL:%.*]] = cir.load [[X_ALLOCA]] : cir.ptr <!s32i>, !s32i
// CHECK-NEXT: [[X_CASTED:%.*]] = cir.cast(integral, [[X_VAL]] : !s32i), !s64i
// CHECK-NEXT: cir.store [[X_CASTED]], [[B_STORAGE0]] : !s64i, cir.ptr <!s64i>
// CHECK-NEXT: [[ONE:%.*]] = cir.const(#cir.int<1> : !s64i) : !s64i
// CHECK-NEXT: [[SE_EL:%.*]] = cir.ptr_stride([[FI_EL]] : !cir.ptr<!ty_22T22>, [[ONE]] : !s64i), !cir.ptr<!ty_22T22>
// CHECK-NEXT: [[A_STORAGE1:%.*]] = cir.get_member [[SE_EL]][0] {name = "a"} : !cir.ptr<!ty_22T22> -> !cir.ptr<!s32i>
// CHECK-NEXT: [[B_STORAGE1:%.*]] = cir.get_member [[SE_EL]][1] {name = "b"} : !cir.ptr<!ty_22T22> -> !cir.ptr<!s64i>
// CHECK-NEXT: cir.return

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

void zero_init(int x) {
  int arr[3] = {x};
}

// CHECK:  cir.func @zero_init
// CHECK:    [[VAR_ALLOC:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK:    %1 = cir.alloca !cir.array<!s32i x 3>, cir.ptr <!cir.array<!s32i x 3>>, ["arr", init] {alignment = 4 : i64}
// CHECK:    [[TEMP:%.*]] = cir.alloca !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>, ["arrayinit.temp", init] {alignment = 8 : i64}
// CHECK:    cir.store %arg0, [[VAR_ALLOC]] : !s32i, cir.ptr <!s32i>
// CHECK:    [[BEGIN:%.*]] = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
// CHECK:    [[VAR:%.*]] = cir.load [[VAR_ALLOC]] : cir.ptr <!s32i>, !s32i
// CHECK:    cir.store [[VAR]], [[BEGIN]] : !s32i, cir.ptr <!s32i>
// CHECK:    [[ONE:%.*]] = cir.const(#cir.int<1> : !s64i) : !s64i
// CHECK:    [[ZERO_INIT_START:%.*]] = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK:    cir.store [[ZERO_INIT_START]], [[TEMP]] : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:    [[SIZE:%.*]] = cir.const(#cir.int<3> : !s64i) : !s64i
// CHECK:    [[END:%.*]] = cir.ptr_stride([[BEGIN]] : !cir.ptr<!s32i>, [[SIZE]] : !s64i), !cir.ptr<!s32i>
// CHECK:    cir.do {
// CHECK:      [[CUR:%.*]] = cir.load [[TEMP]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:      [[FILLER:%.*]] = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:      cir.store [[FILLER]], [[CUR]] : !s32i, cir.ptr <!s32i>
// CHECK:      [[ONE:%.*]] = cir.const(#cir.int<1> : !s64i) : !s64i
// CHECK:      [[NEXT:%.*]] = cir.ptr_stride([[CUR]] : !cir.ptr<!s32i>, [[ONE]] : !s64i), !cir.ptr<!s32i>
// CHECK:      cir.store [[NEXT]], [[TEMP]] : !cir.ptr<!s32i>, cir.ptr <!cir.ptr<!s32i>>
// CHECK:      cir.yield
// CHECK:    } while {
// CHECK:      [[CUR:%.*]] = cir.load [[TEMP]] : cir.ptr <!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:      [[CMP:%.*]] = cir.cmp(ne, [[CUR]], [[END]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK:      cir.condition([[CMP]])
// CHECK:    }
// CHECK:    cir.return
