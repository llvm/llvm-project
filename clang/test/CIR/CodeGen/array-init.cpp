// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef struct {
  int a;
  int b[2];
} A;

int bar() {
  return 42;
}

void foo() {
  A a = {bar(), {}};
}
// CHECK: %[[VAL_0:.*]] = cir.alloca !ty_A, !cir.ptr<!ty_A>, ["a", init]
// CHECK: %[[VAL_1:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["arrayinit.temp", init]
// CHECK: %[[VAL_2:.*]] = cir.get_member %[[VAL_0]][0] {name = "a"} : !cir.ptr<!ty_A> -> !cir.ptr<!s32i>
// CHECK: %[[VAL_3:.*]] = cir.call @_Z3barv() : () -> !s32i
// CHECK: cir.store %[[VAL_3]], %[[VAL_2]] : !s32i, !cir.ptr<!s32i>
// CHECK: %[[VAL_4:.*]] = cir.get_member %[[VAL_0]][1] {name = "b"} : !cir.ptr<!ty_A> -> !cir.ptr<!cir.array<!s32i x 2>>
// CHECK: %[[VAL_5:.*]] = cir.cast(array_to_ptrdecay, %[[VAL_4]] : !cir.ptr<!cir.array<!s32i x 2>>), !cir.ptr<!s32i>
// CHECK: cir.store %[[VAL_5]], %[[VAL_1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %[[VAL_6:.*]] = cir.const #cir.int<2> : !s64i
// CHECK: %[[VAL_7:.*]] = cir.ptr_stride(%[[VAL_5]] : !cir.ptr<!s32i>, %[[VAL_6]] : !s64i), !cir.ptr<!s32i>
// CHECK: cir.do {
// CHECK:     %[[VAL_8:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %[[VAL_9:.*]] = cir.const #cir.int<0> : !s32i
// CHECK:     cir.store %[[VAL_9]], %[[VAL_8]] : !s32i, !cir.ptr<!s32i>
// CHECK:     %[[VAL_10:.*]] = cir.const #cir.int<1> : !s64i
// CHECK:     %[[VAL_11:.*]] = cir.ptr_stride(%[[VAL_8]] : !cir.ptr<!s32i>, %[[VAL_10]] : !s64i), !cir.ptr<!s32i>
// CHECK:     cir.store %[[VAL_11]], %[[VAL_1]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK:     cir.yield
// CHECK: } while {
// CHECK:     %[[VAL_8:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK:     %[[VAL_9:.*]] = cir.cmp(ne, %[[VAL_8]], %[[VAL_7]]) : !cir.ptr<!s32i>, !cir.bool
// CHECK:     cir.condition(%[[VAL_9]])
// CHECK: }