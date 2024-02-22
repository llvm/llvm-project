// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s

// CHECK: !ty_22S22 = !cir.struct<struct "S" {!cir.int<u, 32>, !cir.int<u, 32>, !cir.int<u, 16>, !cir.int<u, 32>} #cir.record.decl.ast>
typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f;
} S;

// CHECK: #bfi_d = #cir.bitfield_info<name = "d", storage_type = !u32i, size = 2, offset = 17, is_signed = true>
// CHECK: #bfi_e = #cir.bitfield_info<name = "e", storage_type = !u16i, size = 15, offset = 0, is_signed = true>

// CHECK: cir.func {{.*@store_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, ["s"]
// CHECK:   [[TMP1:%.*]] = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP0]][2] {name = "e"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u16i>
// CHECK:   [[TMP3:%.*]] = cir.set_bitfield(#bfi_e, [[TMP2]] : !cir.ptr<!u16i>, [[TMP1]] : !s32i) -> !s32i
void store_field() {
  S s;
  s.e = 3;
}

// CHECK: cir.func {{.*@load_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22S22>, cir.ptr <!cir.ptr<!ty_22S22>>, ["s", init]
// CHECK:   [[TMP1:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22S22>>, !cir.ptr<!ty_22S22>
// CHECK:   [[TMP2:%.*]] = cir.get_member [[TMP1]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// CHECK:   [[TMP3:%.*]] = cir.get_bitfield(#bfi_d, [[TMP2]] : !cir.ptr<!u32i>) -> !s32i
int load_field(S* s) {
  return s->d;
}

// CHECK: cir.func {{.*@unOp}}
// CHECK:   [[TMP0:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// CHECK:   [[TMP1:%.*]] = cir.get_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!u32i>) -> !s32i
// CHECK:   [[TMP2:%.*]] = cir.unary(inc, [[TMP1]]) : !s32i, !s32i
// CHECK:   [[TMP3:%.*]] = cir.set_bitfield(#bfi_d, [[TMP0]] : !cir.ptr<!u32i>, [[TMP2]] : !s32i) -> !s32i
void unOp(S* s) {
  s->d++;
}

// CHECK: cir.func {{.*@binOp}}
// CHECK:   [[TMP0:%.*]] = cir.const(#cir.int<42> : !s32i) : !s32i
// CHECK:   [[TMP1:%.*]] = cir.get_member {{.*}}[1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// CHECK:   [[TMP2:%.*]] = cir.get_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!u32i>) -> !s32i
// CHECK:   [[TMP3:%.*]] = cir.binop(or, [[TMP2]], [[TMP0]]) : !s32i
// CHECK:   [[TMP4:%.*]] = cir.set_bitfield(#bfi_d, [[TMP1]] : !cir.ptr<!u32i>, [[TMP3]] : !s32i) -> !s32i
void binOp(S* s) {
   s->d |= 42;
}
