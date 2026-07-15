// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-cir -mmlir -mlir-print-ir-before=cir-cxxabi-lowering %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --check-prefix=CIR,CIR-BEFORE --input-file=%t-before.cir %s
// RUN: FileCheck --check-prefix=CIR,CIR-AFTER --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -Wno-unused-value -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct Empty {};
// CIR-DAG: !rec_Empty = !cir.struct<"Empty" padded {!u8i}>
// LLVMCIR-DAG: %struct.Empty = type { i8 }

struct HasEmpty {
  int size;
  Empty s;
};
// CIR-DAG: !rec_HasEmpty = !cir.struct<"HasEmpty" {!s32i, !rec_Empty}>
// LLVMCIR-DAG: %struct.HasEmpty = type { i32, %struct.Empty }
// OGCG-DAG:    %struct.HasEmpty = type { i32, [4 x i8] }

const HasEmpty globalHE = {1, {}};
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL8globalHE = #cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !rec_Empty}> : !rec_HasEmpty
// LLVMCIR-DAG: @_ZL8globalHE = internal constant %struct.HasEmpty { i32 1, %struct.Empty zeroinitializer }
// OGCG-DAG:    @_ZL8globalHE = internal constant %struct.HasEmpty { i32 1, [4 x i8] undef }

struct HasEmpty2 {
  Empty s;
  int size;
};
// CIR-DAG: !rec_HasEmpty2 = !cir.struct<"HasEmpty2" {!rec_Empty, !s32i}>
// LLVMCIR-DAG: %struct.HasEmpty2 = type { %struct.Empty, i32 }
// OGCG-DAG:    %struct.HasEmpty2 = type { [4 x i8], i32 }

const HasEmpty2 globalHE2 = {{}, 1};
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL9globalHE2 = #cir.const_record<{#cir.zero : !rec_Empty, #cir.int<1> : !s32i}> : !rec_HasEmpty2
// LLVMCIR-DAG: @_ZL9globalHE2 = internal constant %struct.HasEmpty2 { %struct.Empty zeroinitializer, i32 1 }
// OGCG-DAG:    @_ZL9globalHE2 = internal constant %struct.HasEmpty2 { [4 x i8] undef, i32 1 }

// Not referenced enough to be emitted in 'after'.
struct EmptyBase{};
// CIR-BEFORE-DAG: !rec_EmptyBase = !cir.struct<"EmptyBase" padded {!u8i}>

struct Base { int i; };
// CIR-DAG: !rec_Base = !cir.struct<"Base" {!s32i}>
// LLVMCIR-DAG: %struct.Base = type { i32 }
struct D : EmptyBase, Base {};
// CIR-DAG: !rec_D = !cir.struct<"D" {!rec_Base}>
// LLVMCIR-DAG: %struct.D = type { %struct.Base }
int D::* d_i = &D::i;
// CIR-BEFORE-DAG: cir.global external @d_i = #cir.data_member<[0, 0]> : !cir.data_member<!s32i in !rec_D>
// CIR-AFTER-DAG: cir.global external @d_i = #cir.int<0> : !s64i
// LLVM-DAG: @d_i = global i64 0

struct EmptyBase2 { Empty s; };
// CIR-DAG: !rec_EmptyBase2 = !cir.struct<"EmptyBase2" {!rec_Empty}>
// LLVMCIR-DAG: %struct.EmptyBase2 = type { %struct.Empty }
struct Base2 { int i; };
// CIR-DAG: !rec_Base2 = !cir.struct<"Base2" {!s32i}>
// LLVMCIR-DAG: %struct.Base2 = type { i32 }
struct D2 : EmptyBase2, Base2 {};
// CIR-DAG: !rec_D2 = !cir.struct<"D2" {!rec_EmptyBase2, !rec_Base2}>
// LLVMCIR-DAG: %struct.D2 = type { %struct.EmptyBase2, %struct.Base2 }
Empty D2::* d2_s = &D2::s;
// CIR-BEFORE-DAG: cir.global external @d2_s = #cir.data_member<[0, 0]> : !cir.data_member<!rec_Empty in !rec_D2>
// CIR-AFTER-DAG: cir.global external @d2_s = #cir.int<0> : !s64i
// LLVM-DAG: @d2_s = global i64 0
int D2::* d2_i = &D2::i;
// CIR-BEFORE-DAG: cir.global external @d2_i = #cir.data_member<[1, 0]> : !cir.data_member<!s32i in !rec_D2>
// CIR-AFTER-DAG: cir.global external @d2_i = #cir.int<4> : !s64i
// LLVM-DAG: @d2_i = global i64 4

const D globalD = {{}, 1};
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL7globalD = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i}> : !rec_Base}> : !rec_D
// LLVMCIR-DAG: @_ZL7globalD = internal constant %struct.D { %struct.Base { i32 1 } }
// OGCG-DAG:    @_ZL7globalD = internal constant { i32 } { i32 1 }, align 4
const D2 globalD2 = {{}, 1};
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL8globalD2 = #cir.const_record<{#cir.zero : !rec_EmptyBase2, #cir.const_record<{#cir.int<1> : !s32i}> : !rec_Base2}> : !rec_D2
// LLVMCIR-DAG: @_ZL8globalD2 = internal constant %struct.D2 { %struct.EmptyBase2 zeroinitializer, %struct.Base2 { i32 1 } }
// OGCG-DAG:    @_ZL8globalD2 = internal constant { [4 x i8], i32 } { [4 x i8] undef, i32 1 }

struct hasNUA {
  [[no_unique_address]] EmptyBase eb1;
  [[no_unique_address]] EmptyBase eb2;
  [[no_unique_address]] EmptyBase eb3;
  [[no_unique_address]] EmptyBase eb4;
  [[no_unique_address]] EmptyBase eb5;
  [[no_unique_address]] EmptyBase eb6;
  int i;
};
// CIR-DAG: !rec_hasNUA = !cir.struct<"hasNUA" padded {!s32i, !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.hasNUA = type { i32, [4 x i8] }

const hasNUA nua = {{},{},{},{},{},{}, 1};
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL3nua = #cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_hasNUA
// LLVMCIR-DAG: @_ZL3nua = internal constant %struct.hasNUA { i32 1, [4 x i8] zeroinitializer }
// OGCG-DAG:    @_ZL3nua = internal constant %struct.hasNUA { i32 1, [4 x i8] undef }

// A pointer-to-data-member for a no_unique_address empty field has no CIR
// field index (the field isn't laid out), so it is represented by its concrete
// byte offset via #cir.data_member_offset.
EmptyBase hasNUA::* eb1 = &hasNUA::eb1;
// CIR-BEFORE-DAG: cir.global external @eb1 = #cir.data_member_offset<0> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb1 = #cir.int<0> : !s64i
// LLVM-DAG: @eb1 = global i64 0
EmptyBase hasNUA::* eb2 = &hasNUA::eb2;
// CIR-BEFORE-DAG: cir.global external @eb2 = #cir.data_member_offset<1> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb2 = #cir.int<1> : !s64i
// LLVM-DAG: @eb2 = global i64 1
EmptyBase hasNUA::* eb3 = &hasNUA::eb3;
// CIR-BEFORE-DAG: cir.global external @eb3 = #cir.data_member_offset<2> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb3 = #cir.int<2> : !s64i
// LLVM-DAG: @eb3 = global i64 2
EmptyBase hasNUA::* eb4 = &hasNUA::eb4;
// CIR-BEFORE-DAG: cir.global external @eb4 = #cir.data_member_offset<3> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb4 = #cir.int<3> : !s64i
// LLVM-DAG: @eb4 = global i64 3
EmptyBase hasNUA::* eb5 = &hasNUA::eb5;
// CIR-BEFORE-DAG: cir.global external @eb5 = #cir.data_member_offset<4> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb5 = #cir.int<4> : !s64i
// LLVM-DAG: @eb5 = global i64 4
EmptyBase hasNUA::* eb6 = &hasNUA::eb6;
// CIR-BEFORE-DAG: cir.global external @eb6 = #cir.data_member_offset<5> : !cir.data_member<!rec_EmptyBase in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @eb6 = #cir.int<5> : !s64i
// LLVM-DAG: @eb6 = global i64 5
int hasNUA::* nua_i = &hasNUA::i;
// CIR-BEFORE-DAG: cir.global external @nua_i = #cir.data_member<[0]> : !cir.data_member<!s32i in !rec_hasNUA>
// CIR-AFTER-DAG: cir.global external @nua_i = #cir.int<0> : !s64i
// LLVM-DAG: @nua_i = global i64 0

struct FirstField { int a; };
struct HoldsNUA { int b; [[no_unique_address]] EmptyBase e; [[no_unique_address]] EmptyBase e2; };
struct DerivesNUA : FirstField, HoldsNUA {};
EmptyBase DerivesNUA::* nua_in_base = &DerivesNUA::e;
// CIR-BEFORE-DAG: cir.global external @nua_in_base = #cir.data_member_offset<4> : !cir.data_member<!rec_EmptyBase in !rec_DerivesNUA>
// CIR-AFTER-DAG: cir.global external @nua_in_base = #cir.int<4> : !s64i
// LLVM-DAG: @nua_in_base = global i64 4
EmptyBase DerivesNUA::* nua_in_base2 = &DerivesNUA::e2;
// CIR-BEFORE-DAG: cir.global external @nua_in_base2 = #cir.data_member_offset<8> : !cir.data_member<!rec_EmptyBase in !rec_DerivesNUA>
// CIR-AFTER-DAG: cir.global external @nua_in_base2 = #cir.int<8> : !s64i
// LLVM-DAG: @nua_in_base2 = global i64 8

union U1 {
  EmptyBase eb;
};
// Rewrite of the data_member makes this unreferenced 'after'.
// CIR-BEFORE-DAG: !rec_U1 = !cir.union<"U1" {!rec_EmptyBase}>

EmptyBase U1::* u1eb = &U1::eb;
// CIR-BEFORE-DAG: cir.global external @u1eb = #cir.data_member<[0]> : !cir.data_member<!rec_EmptyBase in !rec_U1>
// CIR-AFTER-DAG: cir.global external @u1eb = #cir.int<0> : !s64i
// LLVM-DAG: @u1eb = global i64 0

union U2 {
  int i;
  EmptyBase eb;
};
// Rewrite of the data_member makes this unreferenced 'after'.
// CIR-BEFORE-DAG: !rec_U2 = !cir.union<"U2" {!s32i, !rec_EmptyBase}>

int U2::* u2i = &U2::i;
// CIR-BEFORE-DAG: cir.global external @u2i = #cir.data_member<[0]> : !cir.data_member<!s32i in !rec_U2>
// CIR-AFTER-DAG: cir.global external @u2i = #cir.int<0> : !s64i
// LLVM-DAG: @u2i = global i64 0
EmptyBase U2::* u2eb = &U2::eb;
// CIR-BEFORE-DAG: cir.global external @u2eb = #cir.data_member<[1]> : !cir.data_member<!rec_EmptyBase in !rec_U2>
// CIR-AFTER-DAG: cir.global external @u2eb = #cir.int<0> : !s64i
// LLVM-DAG: @u2eb = global i64 0

union U3 {
  EmptyBase eb;
  int i;
};
// Rewrite of the data_member makes this unreferenced 'after'.
// CIR-BEFORE-DAG: !rec_U3 = !cir.union<"U3" {!rec_EmptyBase, !s32i}>

int U3::* u3i = &U3::i;
// CIR-BEFORE-DAG: cir.global external @u3i = #cir.data_member<[1]> : !cir.data_member<!s32i in !rec_U3>
// CIR-AFTER-DAG: cir.global external @u3i = #cir.int<0> : !s64i
// LLVM-DAG: @u3i = global i64 0
EmptyBase U3::* u3eb = &U3::eb;
// CIR-BEFORE-DAG: cir.global external @u3eb = #cir.data_member<[0]> : !cir.data_member<!rec_EmptyBase in !rec_U3>
// CIR-AFTER-DAG: cir.global external @u3eb = #cir.int<0> : !s64i
// LLVM-DAG: @u3eb = global i64 0

union U4 {
  EmptyBase eb;
  EmptyBase eb2;
  int i;
};
// CIR-BEFORE-DAG: !rec_U4 = !cir.union<"U4" {!rec_EmptyBase, !rec_EmptyBase, !s32i}>

int U4::* u4i = &U4::i;
// CIR-BEFORE-DAG: cir.global external @u4i = #cir.data_member<[2]> : !cir.data_member<!s32i in !rec_U4>
// CIR-AFTER-DAG: cir.global external @u4i = #cir.int<0> : !s64i
// LLVM-DAG: @u4i = global i64 0
EmptyBase U4::* u4eb = &U4::eb;
// CIR-BEFORE-DAG: cir.global external @u4eb = #cir.data_member<[0]> : !cir.data_member<!rec_EmptyBase in !rec_U4>
// CIR-AFTER-DAG: cir.global external @u4eb = #cir.int<0> : !s64i
// LLVM-DAG: @u4eb = global i64 0
EmptyBase U4::* u4eb2 = &U4::eb2;
// CIR-BEFORE-DAG: cir.global external @u4eb2 = #cir.data_member<[1]> : !cir.data_member<!rec_EmptyBase in !rec_U4>
// CIR-AFTER-DAG: cir.global external @u4eb2 = #cir.int<0> : !s64i
// LLVM-DAG: @u4eb2 = global i64 0

union U5 {
  int i;
  EmptyBase eb;
  EmptyBase eb2;
};
// CIR-BEFORE-DAG: !rec_U5 = !cir.union<"U5" {!s32i, !rec_EmptyBase, !rec_EmptyBase}>

int U5::* u5i = &U5::i;
// CIR-BEFORE-DAG: cir.global external @u5i = #cir.data_member<[0]> : !cir.data_member<!s32i in !rec_U5>
// CIR-AFTER-DAG: cir.global external @u5i = #cir.int<0> : !s64i
// LLVM-DAG: @u5i = global i64 0
EmptyBase U5::* u5eb = &U5::eb;
// CIR-BEFORE-DAG: cir.global external @u5eb = #cir.data_member<[1]> : !cir.data_member<!rec_EmptyBase in !rec_U5>
// CIR-AFTER-DAG: cir.global external @u5eb = #cir.int<0> : !s64i
// LLVM-DAG: @u5eb = global i64 0
EmptyBase U5::* u5eb2 = &U5::eb2;
// CIR-BEFORE-DAG: cir.global external @u5eb2 = #cir.data_member<[2]> : !cir.data_member<!rec_EmptyBase in !rec_U5>
// CIR-AFTER-DAG: cir.global external @u5eb2 = #cir.int<0> : !s64i
// LLVM-DAG: @u5eb2 = global i64 0

union U6 {
  [[no_unique_address]]
  EmptyBase eb;
  [[no_unique_address]]
  EmptyBase eb2;
  int i;
};
// CIR-BEFORE-DAG: !rec_U6 = !cir.union<"U6" {!rec_EmptyBase, !rec_EmptyBase, !s32i}>
int U6::* u6i = &U6::i;
// CIR-BEFORE-DAG: cir.global external @u6i = #cir.data_member<[2]> : !cir.data_member<!s32i in !rec_U6>
// CIR-AFTER-DAG: cir.global external @u6i = #cir.int<0> : !s64i
// LLVM-DAG: @u6i = global i64 0

EmptyBase U6::* u6eb = &U6::eb;
// CIR-BEFORE-DAG: cir.global external @u6eb = #cir.data_member_offset<0> : !cir.data_member<!rec_EmptyBase in !rec_U6>
// CIR-AFTER-DAG: cir.global external @u6eb = #cir.int<0> : !s64i
// LLVM-DAG: @u6eb = global i64 0
EmptyBase U6::* u6eb2 = &U6::eb2;
// CIR-BEFORE-DAG: cir.global external @u6eb2 = #cir.data_member_offset<0> : !cir.data_member<!rec_EmptyBase in !rec_U6>
// CIR-AFTER-DAG: cir.global external @u6eb2 = #cir.int<0> : !s64i
// LLVM-DAG: @u6eb2 = global i64 0

union U7 {
  int i;
  [[no_unique_address]]
  EmptyBase eb;
  [[no_unique_address]]
  EmptyBase eb2;
};
// CIR-BEFORE-DAG: !rec_U7 = !cir.union<"U7" {!s32i, !rec_EmptyBase, !rec_EmptyBase}>
int U7::* u7i = &U7::i;
// CIR-BEFORE-DAG: cir.global external @u7i = #cir.data_member<[0]> : !cir.data_member<!s32i in !rec_U7>
// CIR-AFTER-DAG: cir.global external @u7i = #cir.int<0> : !s64i
// LLVM-DAG: @u7i = global i64 0

EmptyBase U7::* u7eb = &U7::eb;
// CIR-BEFORE-DAG: cir.global external @u7eb = #cir.data_member_offset<0> : !cir.data_member<!rec_EmptyBase in !rec_U7>
// CIR-AFTER-DAG: cir.global external @u7eb = #cir.int<0> : !s64i
// LLVM-DAG: @u7eb = global i64 0
EmptyBase U7::* u7eb2 = &U7::eb2;
// CIR-BEFORE-DAG: cir.global external @u7eb2 = #cir.data_member_offset<0> : !cir.data_member<!rec_EmptyBase in !rec_U7>
// CIR-AFTER-DAG: cir.global external @u7eb2 = #cir.int<0> : !s64i
// LLVM-DAG: @u7eb2 = global i64 0

void uses() {
  auto x = &HasEmpty::s;
  auto y = &HasEmpty2::s;

  globalHE.size;
  globalHE2.size;

  globalD.i;
  globalD2.i;

  nua.i;
}
