// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

typedef union {
  int x;
  int y : 4;
  int z : 8;
} demo;

// CIR:  !rec_demo = !cir.record<union "demo" {!s32i, !u8i, !u8i}>
// LLVM: %union.demo = type { i32 }
// OGCG: %union.demo = type { i32 }

typedef union {
  int x;
  int y : 3;
  int   : 0;
  int z : 2;
} zero_bit;

// CIR:  !rec_zero_bit = !cir.record<union "zero_bit" {!s32i, !u8i, !u8i}>
// LLVM: %union.zero_bit = type { i32 }
// OGCG: %union.zero_bit = type { i32 }

demo d;
zero_bit z;

void f() {
    demo d;
    d.x = 1;
    d.y = 2;
    d.z = 0;
}

// CIR: #bfi_y = #cir.bitfield_info<name = "y", storage_type = !u8i, size = 4, offset = 0, is_signed = true>
// CIR: #bfi_z = #cir.bitfield_info<name = "z", storage_type = !u8i, size = 8, offset = 0, is_signed = true>

// CIR:   cir.func no_proto dso_local @f
// CIR:    [[ALLOC:%.*]] = cir.alloca !rec_demo, !cir.ptr<!rec_demo>, ["d"] {alignment = 4 : i64}
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    [[X:%.*]] = cir.get_member [[ALLOC]][0] {name = "x"} : !cir.ptr<!rec_demo> -> !cir.ptr<!s32i>
// CIR:    cir.store align(4) [[ONE]], [[X]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[TWO:%.*]] = cir.const #cir.int<2> : !s32i
// CIR:    [[Y:%.*]] = cir.get_member [[ALLOC]][1] {name = "y"} : !cir.ptr<!rec_demo> -> !cir.ptr<!u8i>
// CIR:    [[SET:%.*]] = cir.set_bitfield align(4) (#bfi_y, [[Y]] : !cir.ptr<!u8i>, [[TWO]] : !s32i) -> !s32i
// CIR:    [[ZERO:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:    [[Z:%.*]] = cir.get_member [[ALLOC]][2] {name = "z"} : !cir.ptr<!rec_demo> -> !cir.ptr<!u8i>
// CIR:    [[SET2:%.*]] = cir.set_bitfield align(4) (#bfi_z, [[Z]] : !cir.ptr<!u8i>, [[ZERO]] : !s32i) -> !s32i
// CIR:    cir.return

// LLVM: define dso_local void @f
// LLVM:   [[ALLOC:%.*]] = alloca %union.demo, i64 1, align 4
// LLVM:   store i32 1, ptr [[ALLOC]], align 4
// LLVM:   [[BFLOAD:%.*]] = load i8, ptr [[ALLOC]], align 4
// LLVM:   [[CLEAR:%.*]] = and i8 [[BFLOAD]], -16
// LLVM:   [[SET:%.*]] = or i8 [[CLEAR]], 2
// LLVM:   store i8 [[SET]], ptr [[ALLOC]], align 4
// LLVM:   store i8 0, ptr [[ALLOC]], align 4

// OGCG: define dso_local void @f
// OGCG:   [[ALLOC:%.*]] = alloca %union.demo, align 4
// OGCG:   store i32 1, ptr [[ALLOC]], align 4
// OGCG:   [[BFLOAD:%.*]] = load i8, ptr [[ALLOC]], align 4
// OGCG:   [[CLEAR:%.*]] = and i8 [[BFLOAD]], -16
// OGCG:   [[SET:%.*]] = or i8 [[CLEAR]], 2
// OGCG:   store i8 [[SET]], ptr [[ALLOC]], align 4
// OGCG:   store i8 0, ptr [[ALLOC]], align 4
