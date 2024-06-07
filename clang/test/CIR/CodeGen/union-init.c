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

// CHECK-DAG: ![[anon0:.*]] = !cir.struct<struct  {!cir.int<u, 32>}>
// CHECK-DAG: ![[anon:.*]] = !cir.struct<struct  {!cir.int<s, 32>}>
// CHECK-DAG: #[[bfi_x:.*]] = #cir.bitfield_info<name = "x", storage_type = !u32i, size = 16, offset = 0, is_signed = true>
// CHECK-DAG: #[[bfi_y:.*]] = #cir.bitfield_info<name = "y", storage_type = !u32i, size = 16, offset = 16, is_signed = true>
// CHECK-DAG: ![[anon1:.*]] = !cir.struct<union "{{.*}}" {!cir.int<u, 32>, !cir.array<!cir.int<u, 8> x 4>}

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

unsigned is_little(void) {
  const union {
    unsigned int u;
    unsigned char c[4];
  } one = {1};
  return one.c[0];
}

// CHECK: cir.func @is_little
// CHECK: %[[VAL_1:.*]] = cir.get_global @is_little.one : !cir.ptr<![[anon0]]>
// CHECK: %[[VAL_2:.*]] = cir.cast(bitcast, %[[VAL_1]] : !cir.ptr<![[anon0]]>), !cir.ptr<![[anon1]]>
// CHECK: %[[VAL_3:.*]] = cir.get_member %[[VAL_2]][1] {name = "c"} : !cir.ptr<![[anon1]]> -> !cir.ptr<!cir.array<!u8i x 4>>