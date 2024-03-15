// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s
// XFAIL: *

typedef union {
  int value;
  struct {
    int x : 16;
    int y : 16;
  };
} A;

void foo(int x) {
  A a = {.x = x};
}

// CHECK:  cir.func @foo(%arg0: !s32i loc({{.*}}))
// CHECK:    [[TMP0:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK:    [[TMP1:%.*]] = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["a", init] {alignment = 4 : i64}
// CHECK:    cir.store %arg0, [[TMP0]] : !s32i, cir.ptr <!s32i>
// CHECK:    [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = ""} : !cir.ptr<!ty_22A22> -> !cir.ptr<!ty_22anon2E122>
// CHECK:    [[TMP3:%.*]] = cir.cast(bitcast, [[TMP2]] : !cir.ptr<!ty_22anon2E122>), !cir.ptr<!u32i>
// CHECK:    [[TMP4:%.*]] = cir.load [[TMP0]] : cir.ptr <!s32i>, !s32i
// CHECK:    [[TMP5:%.*]] = cir.cast(integral, [[TMP4]] : !s32i), !u32i
// CHECK:    [[TMP6:%.*]] = cir.load [[TMP3]] : cir.ptr <!u32i>, !u32i
// CHECK:    [[TMP7:%.*]] = cir.const(#cir.int<65535> : !u32i) : !u32i
// CHECK:    [[TMP8:%.*]] = cir.binop(and, [[TMP5]], [[TMP7]]) : !u32i
// CHECK:    [[TMP9:%.*]] = cir.const(#cir.int<4294901760> : !u32i) : !u32i
// CHECK:    [[TMP10:%.*]] = cir.binop(and, [[TMP6]], [[TMP9]]) : !u32i
// CHECK:    [[TMP11:%.*]] = cir.binop(or, [[TMP10]], [[TMP8]]) : !u32i
// CHECK:    cir.store [[TMP11]], [[TMP3]] : !u32i, cir.ptr <!u32i>
// CHECK:    [[TMP12:%.*]] = cir.cast(bitcast, [[TMP2]] : !cir.ptr<!ty_22anon2E122>), !cir.ptr<!u32i>
// CHECK:    [[TMP13:%.*]] = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK:    [[TMP14:%.*]] = cir.cast(integral, [[TMP13]] : !s32i), !u32i
// CHECK:    [[TMP15:%.*]] = cir.load [[TMP12]] : cir.ptr <!u32i>, !u32i
// CHECK:    [[TMP16:%.*]] = cir.const(#cir.int<65535> : !u32i) : !u32i
// CHECK:    [[TMP17:%.*]] = cir.binop(and, [[TMP14]], [[TMP16]]) : !u32i
// CHECK:    [[TMP18:%.*]] = cir.const(#cir.int<16> : !u32i) : !u32i
// CHECK:    [[TMP19:%.*]] = cir.shift(left, [[TMP17]] : !u32i, [[TMP18]] : !u32i) -> !u32i
// CHECK:    [[TMP20:%.*]] = cir.const(#cir.int<65535> : !u32i) : !u32i
// CHECK:    [[TMP21:%.*]] = cir.binop(and, [[TMP15]], [[TMP20]]) : !u32i
// CHECK:    [[TMP22:%.*]] = cir.binop(or, [[TMP21]], [[TMP19]]) : !u32i
// CHECK:    cir.store [[TMP22]], [[TMP12]] : !u32i, cir.ptr <!u32i>
// CHECK:    cir.return
