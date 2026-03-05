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

// CHECK-DAG: !rec_anon_struct = !cir.record<struct  {!u8i, !u8i, !u8i, !u8i, !s32i}>
// CHECK-DAG: !rec_T = !cir.record<struct "T" {!u32i, !s32i} #cir.record.decl.ast>
// CHECK-DAG: !rec_anon_struct1 = !cir.record<struct  {!u8i, !cir.array<!u8i x 3>, !u8i, !u8i, !u8i, !u8i}>
// CHECK-DAG: #bfi_Z = #cir.bitfield_info<name = "Z", storage_type = !u32i, size = 9, offset = 11, is_signed = true>

struct T GV = { 1, 5, 26, 42 };
// CHECK: cir.global external @GV = #cir.const_record<{#cir.int<161> : !u8i, #cir.int<208> : !u8i, #cir.int<0> : !u8i,  #cir.zero : !u8i, #cir.int<42> : !s32i}> : !rec_anon_struct

// check padding is used (const array of zeros)
struct Inner var = { 1, 0, 1, 21};
// CHECK: cir.global external @var = #cir.const_record<{#cir.int<5> : !u8i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 3>, #cir.int<21> : !u8i, #cir.int<0> : !u8i, #cir.int<0> : !u8i, #cir.int<0> : !u8i}> : !rec_anon_struct1


// CHECK: cir.func {{.*@getZ()}}
// CHECK:   %1 = cir.get_global @GV : !cir.ptr<!rec_anon_struct>
// CHECK:   %2 = cir.cast bitcast %1 : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CHECK:   %3 = cir.get_member %2[0] {name = "Z"} : !cir.ptr<!rec_T> -> !cir.ptr<!u32i>
// CHECK:   %4 = cir.get_bitfield align(4) (#bfi_Z, %3 : !cir.ptr<!u32i>) -> !s32i
int getZ() {
  return GV.Z;
}

// check the type used is the type of T struct for plain field
// CHECK:  cir.func {{.*@getW()}}
// CHECK:    %1 = cir.get_global @GV : !cir.ptr<!rec_anon_struct>
// CHECK:    %2 = cir.cast bitcast %1 : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_T>
// CHECK:    %3 = cir.get_member %2[1] {name = "W"} : !cir.ptr<!rec_T> -> !cir.ptr<!s32i>
int getW() {
  return GV.W;
}

