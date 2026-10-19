// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-layout-compatibility=microsoft -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-layout-compatibility=microsoft -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fms-layout-compatibility=microsoft -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct s1 {
  int       f32 : 2;
  long long f64 : 30;
} s1;

// CIR-DAG: !rec_s1 = !cir.struct<"s1" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 2>]>, bitfield !cir.bitfield<!s64i, [#cir.bitfield_decl<!s64i, 30>]>}>
// LLVM-DAG: %struct.s1 = type { i32, i64 }

struct s2 {
    int a : 24;
    char b;
    int c : 30;
} Clip;

// CIR-DAG: !rec_s2 = !cir.struct<"s2" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 24>]>, data !s8i, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 30>]>}>
// LLVM-DAG: %struct.s2 = type { i32, i8, i32 }

struct s3 {
    int a : 18;
    int   :  0;
    int c : 14;
} zero_bit;

// CIR-DAG:  !rec_s3 = !cir.struct<"s3" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 18>]>, empty !cir.bitfield<[#cir.bitfield_decl<!s32i, 0, unnamed>]>, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 14>]>}>
// LLVM-DAG: %struct.s3 = type { i32, i32 }

#pragma pack (push,1)

struct Inner {
  unsigned int A :  1;
  unsigned int B :  1;
  unsigned int C :  1;
  unsigned int D : 30;
} Inner;

#pragma pack (pop)

// CIR-DAG: !rec_Inner = !cir.struct<"Inner" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!u32i, 1>, #cir.bitfield_decl<!u32i, 1>, #cir.bitfield_decl<!u32i, 1>]>, bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!u32i, 30>]>}>
// LLVM-DAG: %struct.Inner = type { i32, i32 }

#pragma pack(push, 1)

union HEADER {
  struct A {
    int                                         :  3;  // Bits 2:0
    int a                                       :  9;  // Bits 11:3
    int                                         :  12;  // Bits 23:12
    int b                                       :  17;  // Bits 40:24
    int                                         :  7;  // Bits 47:41
    int c                                       :  4;  // Bits 51:48
    int                                         :  4;  // Bits 55:52
    int d                                       :  3;  // Bits 58:56
    int                                         :  5;  // Bits 63:59
  } Bits;
} HEADER;

#pragma pack(pop)

// CIR-DAG: !rec_A = !cir.struct<"A" {bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3, unnamed>, #cir.bitfield_decl<!s32i, 9>, #cir.bitfield_decl<!s32i, 12, unnamed>]>, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 17>, #cir.bitfield_decl<!s32i, 7, unnamed>, #cir.bitfield_decl<!s32i, 4>, #cir.bitfield_decl<!s32i, 4, unnamed>]>, bitfield !cir.bitfield<!s32i, [#cir.bitfield_decl<!s32i, 3>, #cir.bitfield_decl<!s32i, 5, unnamed>]>}>
// CIR-DAG: !rec_HEADER = !cir.union<"HEADER" {data !rec_A}>
// LLVM-DAG: %struct.A = type { i32, i32, i32 }
// LLVM-DAG: %union.HEADER = type { %struct.A }
