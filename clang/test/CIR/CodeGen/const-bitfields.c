// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o - 2>&1 | FileCheck %s

struct T {
  int X : 5;
  int Y : 6;
  int Z : 9;
  int W;  
};

struct Inner {
  unsigned a :  1;
  unsigned b :  1;
  unsigned c :  1;
  unsigned d : 30;
};

// CHECK: !ty_22T22 = !cir.struct<struct "T" {!cir.int<u, 32>, !cir.int<s, 32>} #cir.record.decl.ast>
// CHECK: !ty_anon_struct = !cir.struct<struct  {!cir.int<u, 8>, !cir.int<u, 8>, !cir.int<u, 8>, !cir.int<s, 32>}>
// CHECK: #bfi_Z = #cir.bitfield_info<name = "Z", storage_type = !u32i, size = 9, offset = 11, is_signed = true>
// CHECK: !ty_anon_struct1 = !cir.struct<struct  {!cir.int<u, 8>, !cir.array<!cir.int<u, 8> x 3>, !cir.int<u, 8>, !cir.int<u, 8>, !cir.int<u, 8>, !cir.int<u, 8>}>

struct T GV = { 1, 5, 256, 42 };
// CHECK: cir.global external @GV = #cir.const_struct<{#cir.int<161> : !u8i, #cir.int<0> : !u8i, #cir.int<8> : !u8i, #cir.int<42> : !s32i}> : !ty_anon_struct

// check padding is used (const array of zeros)
struct Inner var = { 1, 0, 1, 21};
// CHECK: cir.global external @var = #cir.const_struct<{#cir.int<5> : !u8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>, #cir.int<21> : !u8i, #cir.int<0> : !u8i, #cir.int<0> : !u8i, #cir.int<0> : !u8i}> : !ty_anon_struct1


// CHECK: cir.func {{.*@getZ()}}
// CHECK:   %1 = cir.get_global @GV : cir.ptr <!ty_anon_struct>
// CHECK:   %2 = cir.cast(bitcast, %1 : !cir.ptr<!ty_anon_struct>), !cir.ptr<!ty_22T22>
// CHECK:   %3 = cir.cast(bitcast, %2 : !cir.ptr<!ty_22T22>), !cir.ptr<!u32i>
// CHECK:   %4 = cir.get_bitfield(#bfi_Z, %3 : !cir.ptr<!u32i>) -> !s32i
int getZ() {
  return GV.Z;
}

// check the type used is the type of T struct for plain field
// CHECK:  cir.func {{.*@getW()}}
// CHECK:    %1 = cir.get_global @GV : cir.ptr <!ty_anon_struct>
// CHECK:    %2 = cir.cast(bitcast, %1 : !cir.ptr<!ty_anon_struct>), !cir.ptr<!ty_22T22>
// CHECK:    %3 = cir.get_member %2[1] {name = "W"} : !cir.ptr<!ty_22T22> -> !cir.ptr<!s32i>
int getW() {
  return GV.W;
}

