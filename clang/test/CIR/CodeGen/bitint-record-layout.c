// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll 
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

union JustBIUnion {
  _BitInt(65) bi;
};
// CIR-DAG: !rec_JustBIUnion = !cir.union<"JustBIUnion" {data !cir.int<s, 65, bitint>}>
// LLVM-DAG: %union.JustBIUnion = type { i128 }

struct SmallerUnionMem { long long a; int b; };
// CIR-DAG: !rec_SmallerUnionMem = !cir.struct<"SmallerUnionMem" {data !s64i, data !s32i}>
// LLVM-DAG: %struct.SmallerUnionMem = type { i64, i32 }

union BIUnion {
  _BitInt(65) bi;
  struct SmallerUnionMem m;
};
// CIR-DAG: !rec_BIUnion = !cir.union<"BIUnion" {data !cir.int<s, 65, bitint>, data !rec_SmallerUnionMem}>
// LLVM-DAG: %union.BIUnion = type { i128 }

union BIUnionArr {
  _BitInt(65) bi;
  char arr[20];
};
// CIR-DAG: !rec_BIUnionArr = !cir.union<"BIUnionArr" packed {data !cir.int<s, 65, bitint>, data !cir.array<!s8i x 20>}, padding = {!cir.array<!u8i x 8>}>
// LLVM-DAG: %union.BIUnionArr = type <{ i128, [8 x i8] }>

struct First65 {
  _BitInt(65) bi;
  int i;
};
// CIR-DAG: !rec_First65 = !cir.struct<"First65" packed {data !cir.int<s, 65, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First65 = type <{ i128, i32, [4 x i8] }>

struct Middle65 {
  char c;
  _BitInt(65) bi;
  int i;
};
// CIR-DAG: !rec_Middle65 = !cir.struct<"Middle65" packed {data !s8i, pad !cir.array<!u8i x 7>, data !cir.int<s, 65, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle65 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

struct Last65 {
  int i;
  _BitInt(65) bi;
};
// CIR-DAG: !rec_Last65 = !cir.struct<"Last65" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.int<s, 65, bitint>}>
// LLVM-DAG: %struct.Last65 = type <{ i32, [4 x i8], i128 }>

struct First127 {
  _BitInt(127) bi;
  int i;
};
// CIR-DAG: !rec_First127 = !cir.struct<"First127" packed {data !cir.int<s, 127, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First127 = type <{ i128, i32, [4 x i8] }>

struct Middle127 {
  char c;
  _BitInt(127) bi;
  int i;
};
// CIR-DAG: !rec_Middle127 = !cir.struct<"Middle127" packed {data !s8i, pad !cir.array<!u8i x 7>, data !cir.int<s, 127, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle127 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

struct Last127 {
  int i;
  _BitInt(127) bi;
};
// CIR-DAG: !rec_Last127 = !cir.struct<"Last127" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.int<s, 127, bitint>}>
// LLVM-DAG: %struct.Last127 = type <{ i32, [4 x i8], i128 }>

struct First128 {
  _BitInt(128) bi;
  int i;
};
// CIR-DAG: !rec_First128 = !cir.struct<"First128" packed {data !s128i_bitint, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First128 = type <{ i128, i32, [4 x i8] }>

struct Middle128 {
  char c;
  _BitInt(128) bi;
  int i;
};
// CIR-DAG: !rec_Middle128 = !cir.struct<"Middle128" packed {data !s8i, pad !cir.array<!u8i x 7>, data !s128i_bitint, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle128 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

struct Last128 {
  int i;
  _BitInt(128) bi;
};
// CIR-DAG: !rec_Last128 = !cir.struct<"Last128" packed {data !s32i, pad !cir.array<!u8i x 4>, data !s128i_bitint}>
// LLVM-DAG: %struct.Last128 = type <{ i32, [4 x i8], i128 }>

struct ArrMem {
  int i;
  _BitInt(128) bi[2];
};
// CIR-DAG: !rec_ArrMem = !cir.struct<"ArrMem" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.array<!s128i_bitint x 2>}>
// LLVM-DAG: %struct.ArrMem = type <{ i32, [4 x i8], [2 x i128] }>

struct ArrMem65 { 
  int i;
  _BitInt(65) bi[2];
};
// CIR-DAG: !rec_ArrMem65 = !cir.struct<"ArrMem65" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.array<!cir.int<s, 65, bitint> x 2>}>
// LLVM-DAG: %struct.ArrMem65 = type <{ i32, [4 x i8], [2 x i128] }>

struct Inner {
  int i;
  _BitInt(128) bi;
};
// CIR-DAG: !rec_Inner = !cir.struct<"Inner" packed {data !s32i, pad !cir.array<!u8i x 4>, data !s128i_bitint}>
// LLVM-DAG: %struct.Inner = type <{ i32, [4 x i8], i128 }>
struct Outer {
  char c;
  struct Inner inner;
};
// CIR-DAG: !rec_Outer = !cir.struct<"Outer" {data !s8i, pad !cir.array<!u8i x 7>, data !rec_Inner}>
// LLVM-DAG: %struct.Outer = type { i8, [7 x i8], %struct.Inner }

struct Inner2 {
  int i;
  _BitInt(128) bi;
};
// CIR-DAG: !rec_Inner2 = !cir.struct<"Inner2" packed {data !s32i, pad !cir.array<!u8i x 4>, data !s128i_bitint}>
// LLVM-DAG: %struct.Inner2 = type <{ i32, [4 x i8], i128 }>

struct Outer2 {
  char c;
  struct Inner2 inner;
  short s;
};
// CIR-DAG: !rec_Outer2 = !cir.struct<"Outer2" {data !s8i, pad !cir.array<!u8i x 7>, data !rec_Inner2, data !s16i, pad !cir.array<!u8i x 6>}>
// LLVM-DAG: %struct.Outer2 = type { i8, [7 x i8], %struct.Inner2, i16, [6 x i8] }


union JustBIUnion jbiu = { .bi = 54321 };
// CIR-DAG: cir.global external @jbiu = #cir.const_record<{#cir.int<54321> : !cir.int<s, 65, bitint>}> : !rec_JustBIUnion {alignment = 8 : i64}
// LLVM-DAG: @jbiu = global %union.JustBIUnion { i128 54321 }, align 8
union BIUnion biunion = { .bi = 12345 };
// CIR-DAG: cir.global external @biunion = #cir.const_record<{#cir.int<12345> : !cir.int<s, 65, bitint>}> : !rec_BIUnion {alignment = 8 : i64}
// LLVM-DAG: @biunion = global %union.BIUnion { i128 12345 }, align 8

union BIUnionArr biuarr = { .arr = { 'a', 'b', 'c' } };
// CIR-DAG: cir.global external @biuarr = #cir.const_record<{#cir.const_array<[#cir.int<97> : !s8i, #cir.int<98> : !s8i, #cir.int<99> : !s8i], trailing_zeros> : !cir.array<!s8i x 20>}> : !rec_BIUnionArr {alignment = 8 : i64}
// Classic-codegen represents the constant by splitting the init part of
// the string and the zeros separately, plus not as a string.  Else this is effectively the same.
// LLVMCIR-DAG: @biuarr = global <{ [20 x i8], [4 x i8] }> <{ [20 x i8] c"abc\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", [4 x i8] zeroinitializer }>, align 8
// OGCG-DAG:    @biuarr = global { <{ i8, i8, i8, [17 x i8] }>, [4 x i8] } { <{ i8, i8, i8, [17 x i8] }> <{ i8 97, i8 98, i8 99, [17 x i8] zeroinitializer }>, [4 x i8] zeroinitializer }, align 8

struct First65 f65[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @f65 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !cir.int<s, 65, bitint>, #cir.int<222> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First65, #cir.const_record<{#cir.int<3> : !cir.int<s, 65, bitint>, #cir.int<444> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First65]> : !cir.array<!rec_First65 x 2> {alignment = 16 : i64}
// LLVM-DAG: @f65 = global [2 x %struct.First65] [%struct.First65 <{ i128 1, i32 222, [4 x i8] zeroinitializer }>, %struct.First65 <{ i128 3, i32 444, [4 x i8] zeroinitializer }>], align 16
struct Middle65 m65[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @m65 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<222> : !cir.int<s, 65, bitint>, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle65, #cir.const_record<{#cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<444> : !cir.int<s, 65, bitint>, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle65]> : !cir.array<!rec_Middle65 x 2> {alignment = 16 : i64}
// LLVM-DAG: @m65 = global [2 x %struct.Middle65] [%struct.Middle65 <{ i8 1, [7 x i8] zeroinitializer, i128 222, i32 0, [4 x i8] zeroinitializer }>, %struct.Middle65 <{ i8 3, [7 x i8] zeroinitializer, i128 444, i32 0, [4 x i8] zeroinitializer }>], align 16
struct Last65 l65[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @l65 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<222> : !cir.int<s, 65, bitint>}> : !rec_Last65, #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<444> : !cir.int<s, 65, bitint>}> : !rec_Last65]> : !cir.array<!rec_Last65 x 2> {alignment = 16 : i64}
// LLVM-DAG: @l65 = global [2 x %struct.Last65] [%struct.Last65 <{ i32 1, [4 x i8] zeroinitializer, i128 222 }>, %struct.Last65 <{ i32 3, [4 x i8] zeroinitializer, i128 444 }>], align 16

struct First127 f127[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @f127 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !cir.int<s, 127, bitint>, #cir.int<222> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First127, #cir.const_record<{#cir.int<3> : !cir.int<s, 127, bitint>, #cir.int<444> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First127]> : !cir.array<!rec_First127 x 2> {alignment = 16 : i64}
// LLVM-DAG: @f127 = global [2 x %struct.First127] [%struct.First127 <{ i128 1, i32 222, [4 x i8] zeroinitializer }>, %struct.First127 <{ i128 3, i32 444, [4 x i8] zeroinitializer }>], align 16
struct Middle127 m127[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @m127 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<222> : !cir.int<s, 127, bitint>, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle127, #cir.const_record<{#cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<444> : !cir.int<s, 127, bitint>, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle127]> : !cir.array<!rec_Middle127 x 2> {alignment = 16 : i64}
// LLVM-DAG: @m127 = global [2 x %struct.Middle127] [%struct.Middle127 <{ i8 1, [7 x i8] zeroinitializer, i128 222, i32 0, [4 x i8] zeroinitializer }>, %struct.Middle127 <{ i8 3, [7 x i8] zeroinitializer, i128 444, i32 0, [4 x i8] zeroinitializer }>], align 16
struct Last127 l127[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @l127 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<222> : !cir.int<s, 127, bitint>}> : !rec_Last127, #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<444> : !cir.int<s, 127, bitint>}> : !rec_Last127]> : !cir.array<!rec_Last127 x 2> {alignment = 16 : i64}
// LLVM-DAG: @l127 = global [2 x %struct.Last127] [%struct.Last127 <{ i32 1, [4 x i8] zeroinitializer, i128 222 }>, %struct.Last127 <{ i32 3, [4 x i8] zeroinitializer, i128 444 }>], align 16

struct First128 f128[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @f128 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s128i_bitint, #cir.int<222> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First128, #cir.const_record<{#cir.int<3> : !s128i_bitint, #cir.int<444> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_First128]> : !cir.array<!rec_First128 x 2> {alignment = 16 : i64}
// LLVM-DAG: @f128 = global [2 x %struct.First128] [%struct.First128 <{ i128 1, i32 222, [4 x i8] zeroinitializer }>, %struct.First128 <{ i128 3, i32 444, [4 x i8] zeroinitializer }>], align 16
struct Middle128 m128[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @m128 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<222> : !s128i_bitint, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle128, #cir.const_record<{#cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.int<444> : !s128i_bitint, #cir.int<0> : !s32i, #cir.zero : !cir.array<!u8i x 4>}> : !rec_Middle128]> : !cir.array<!rec_Middle128 x 2> {alignment = 16 : i64}
// LLVM-DAG: @m128 = global [2 x %struct.Middle128] [%struct.Middle128 <{ i8 1, [7 x i8] zeroinitializer, i128 222, i32 0, [4 x i8] zeroinitializer }>, %struct.Middle128 <{ i8 3, [7 x i8] zeroinitializer, i128 444, i32 0, [4 x i8] zeroinitializer }>], align 16
struct Last128 l128[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @l128 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<222> : !s128i_bitint}> : !rec_Last128, #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<444> : !s128i_bitint}> : !rec_Last128]> : !cir.array<!rec_Last128 x 2> {alignment = 16 : i64}
// LLVM-DAG: @l128 = global [2 x %struct.Last128] [%struct.Last128 <{ i32 1, [4 x i8] zeroinitializer, i128 222 }>, %struct.Last128 <{ i32 3, [4 x i8] zeroinitializer, i128 444 }>], align 16

struct ArrMem arrMem[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @arrMem = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<222> : !s128i_bitint], trailing_zeros> : !cir.array<!s128i_bitint x 2>}> : !rec_ArrMem, #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<444> : !s128i_bitint], trailing_zeros> : !cir.array<!s128i_bitint x 2>}> : !rec_ArrMem]> : !cir.array<!rec_ArrMem x 2> {alignment = 16 : i64}
// LLVM-DAG: @arrMem = global [2 x %struct.ArrMem] [%struct.ArrMem <{ i32 1, [4 x i8] zeroinitializer, [2 x i128] [i128 222, i128 0] }>, %struct.ArrMem <{ i32 3, [4 x i8] zeroinitializer, [2 x i128] [i128 444, i128 0] }>], align 16

struct ArrMem65 arrMem65[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @arrMem65 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<222> : !cir.int<s, 65, bitint>], trailing_zeros> : !cir.array<!cir.int<s, 65, bitint> x 2>}> : !rec_ArrMem65, #cir.const_record<{#cir.int<3> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.const_array<[#cir.int<444> : !cir.int<s, 65, bitint>], trailing_zeros> : !cir.array<!cir.int<s, 65, bitint> x 2>}> : !rec_ArrMem65]> : !cir.array<!rec_ArrMem65 x 2> {alignment = 16 : i64}
// LLVM-DAG: @arrMem65 = global [2 x %struct.ArrMem65] [%struct.ArrMem65 <{ i32 1, [4 x i8] zeroinitializer, [2 x i128] [i128 222, i128 0] }>, %struct.ArrMem65 <{ i32 3, [4 x i8] zeroinitializer, [2 x i128] [i128 444, i128 0] }>], align 16

struct Outer nestedArr[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @nestedArr = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.const_record<{#cir.int<222> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<0> : !s128i_bitint}> : !rec_Inner}> : !rec_Outer, #cir.const_record<{#cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.const_record<{#cir.int<444> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<0> : !s128i_bitint}> : !rec_Inner}> : !rec_Outer]> : !cir.array<!rec_Outer x 2> {alignment = 16 : i64}
// LLVM-DAG: @nestedArr = global [2 x %struct.Outer] [%struct.Outer { i8 1, [7 x i8] zeroinitializer, %struct.Inner <{ i32 222, [4 x i8] zeroinitializer, i128 0 }> }, %struct.Outer { i8 3, [7 x i8] zeroinitializer, %struct.Inner <{ i32 444, [4 x i8] zeroinitializer, i128 0 }> }], align 16
struct Outer2 nestedArr2[2] = {{1, 222}, {3, 444}};
// CIR-DAG: cir.global external @nestedArr2 = #cir.const_array<[#cir.const_record<{#cir.int<1> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.const_record<{#cir.int<222> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<0> : !s128i_bitint}> : !rec_Inner2, #cir.int<0> : !s16i, #cir.zero : !cir.array<!u8i x 6>}> : !rec_Outer2, #cir.const_record<{#cir.int<3> : !s8i, #cir.zero : !cir.array<!u8i x 7>, #cir.const_record<{#cir.int<444> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<0> : !s128i_bitint}> : !rec_Inner2, #cir.int<0> : !s16i, #cir.zero : !cir.array<!u8i x 6>}> : !rec_Outer2]> : !cir.array<!rec_Outer2 x 2> {alignment = 16 : i64}
// LLVM-DAG: @nestedArr2 = global [2 x %struct.Outer2] [%struct.Outer2 { i8 1, [7 x i8] zeroinitializer, %struct.Inner2 <{ i32 222, [4 x i8] zeroinitializer, i128 0 }>, i16 0, [6 x i8] zeroinitializer }, %struct.Outer2 { i8 3, [7 x i8] zeroinitializer, %struct.Inner2 <{ i32 444, [4 x i8] zeroinitializer, i128 0 }>, i16 0, [6 x i8] zeroinitializer }], align 16

_BitInt(128) get_bi(void) { return l128[1].bi; }
// CIR-LABEL: cir.func no_inline dso_local @get_bi() -> !s128i_bitint
// CIR-NEXT: %[[RET_ALLOC:.*]] = cir.alloca "__retval" align(8) : !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: %[[GET_GLOB:.*]] = cir.get_global @l128 : !cir.ptr<!cir.array<!rec_Last128 x 2>>
// CIR-NEXT: %[[ARR_GEP:.*]] = cir.get_element %[[GET_GLOB]][%[[ONE]] : !s64i] : !cir.ptr<!cir.array<!rec_Last128 x 2>> -> !cir.ptr<!rec_Last128>
// CIR-NEXT: %[[GET_BI:.*]] = cir.get_member %[[ARR_GEP]][2] {name = "bi"} : !cir.ptr<!rec_Last128> -> !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[LOAD_BI:.*]] = cir.load align(8) %[[GET_BI]] : !cir.ptr<!s128i_bitint>, !s128i_bitint
// CIR-NEXT: cir.store %[[LOAD_BI]], %[[RET_ALLOC]] : !s128i_bitint, !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOC]] : !cir.ptr<!s128i_bitint>, !s128i_bitint
// CIR-NEXT: cir.return %[[RET_LOAD]] : !s128i_bitint
// LLVM-LABEL: define dso_local i128 @get_bi()
// LLVM: load i128, ptr getelementptr inbounds nuw (i8, ptr @l128, i64 32), align 8

_BitInt(128) get_bi2(void) { return arrMem[1].bi[1]; }
// CIR-LABEL: cir.func no_inline dso_local @get_bi2() -> !s128i_bitint attributes {"cir.target-features" = "+cx8,+mmx,+sse,+sse2,+x87", nothrow} {
// CIR-NEXT: %[[RET_ALLOC:.*]] = cir.alloca "__retval" align(8) : !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: %[[ONE_2:.*]] = cir.const #cir.int<1> : !s64i
// CIR-NEXT: %[[GET_GLOB:.*]] = cir.get_global @arrMem : !cir.ptr<!cir.array<!rec_ArrMem x 2>>
// CIR-NEXT: %[[ARR_GEP:.*]] = cir.get_element %[[GET_GLOB]][%[[ONE_2]] : !s64i] : !cir.ptr<!cir.array<!rec_ArrMem x 2>> -> !cir.ptr<!rec_ArrMem>
// CIR-NEXT: %[[GET_BI_ARR:.*]] = cir.get_member %[[ARR_GEP]][2] {name = "bi"} : !cir.ptr<!rec_ArrMem> -> !cir.ptr<!cir.array<!s128i_bitint x 2>>
// CIR-NEXT: %[[GET_BI_ELT:.*]] = cir.get_element %[[GET_BI_ARR]][%[[ONE]] : !s64i] : !cir.ptr<!cir.array<!s128i_bitint x 2>> -> !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[LOAD_BI:.*]] = cir.load align(8) %[[GET_BI_ELT]] : !cir.ptr<!s128i_bitint>, !s128i_bitint
// CIR-NEXT: cir.store %[[LOAD_BI]], %[[RET_ALLOC]] : !s128i_bitint, !cir.ptr<!s128i_bitint>
// CIR-NEXT: %[[RET_LOAD:.*]] = cir.load %[[RET_ALLOC]] : !cir.ptr<!s128i_bitint>, !s128i_bitint
// CIR-NEXT: cir.return %[[RET_LOAD]] : !s128i_bitint
// LLVM-LABEL: define dso_local i128 @get_bi2()
// LLVM: load i128, ptr getelementptr inbounds nuw (i8, ptr @arrMem, i64 64), align 8


void force_emit() {
  union BIUnionArr b;
  struct SmallerUnionMem su;
}

