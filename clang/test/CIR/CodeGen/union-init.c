// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir %s -o - | FileCheck %s

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

// CHECK: ![[anon:.*]] = !cir.struct<struct  {!cir.int<s, 32>}>
// CHECK: #[[bfi_x:.*]] = #cir.bitfield_info<name = "x", storage_type = !u32i, size = 16, offset = 0, is_signed = true>
// CHECK: #[[bfi_y:.*]] = #cir.bitfield_info<name = "y", storage_type = !u32i, size = 16, offset = 16, is_signed = true>

// CHECK-LABEL:   cir.func @foo(
// CHECK:  %[[VAL_1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CHECK:  %[[VAL_2:.*]] = cir.alloca !ty_22A22, !cir.ptr<!ty_22A22>, ["a", init] {alignment = 4 : i64}
// CHECK:  cir.store {{.*}}, %[[VAL_1]] : !s32i, !cir.ptr<!s32i>
// CHECK:  %[[VAL_3:.*]] = cir.get_member %[[VAL_2]][1] {name = ""} : !cir.ptr<!ty_22A22> -> !cir.ptr<!ty_22anon2E122>
// CHECK:  %[[VAL_4:.*]] = cir.cast(bitcast, %[[VAL_3]] : !cir.ptr<!ty_22anon2E122>), !cir.ptr<!u32i>
// CHECK:  %[[VAL_5:.*]] = cir.load %[[VAL_1]] : !cir.ptr<!s32i>, !s32i
// CHECK:  %[[VAL_6:.*]] = cir.set_bitfield(#[[bfi_x]], %[[VAL_4]] : !cir.ptr<!u32i>, %[[VAL_5]] : !s32i) -> !s32i
// CHECK:  %[[VAL_7:.*]] = cir.cast(bitcast, %[[VAL_3]] : !cir.ptr<!ty_22anon2E122>), !cir.ptr<!u32i>
// CHECK:  %[[VAL_8:.*]] = cir.const #cir.int<0> : !s32i
// CHECK:  %[[VAL_9:.*]] = cir.set_bitfield(#[[bfi_y]], %[[VAL_7]] : !cir.ptr<!u32i>, %[[VAL_8]] : !s32i) -> !s32i
// CHECK:  cir.return

union { int i; float f; } u = { };
// CHECK: cir.global external @u = #cir.zero : ![[anon]]