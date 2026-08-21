// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

struct Last65 {
  int i;
  _BitInt(65) bi;
};
// CIR-DAG: !rec_Last65 = !cir.struct<"Last65" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.int<s, 65, bitint>}>
// LLVMCIR-DAG: %struct.Last65 = type <{ i32, [4 x i8], i128 }>

struct Last127 {
  int i;
  _BitInt(127) bi;
};
// CIR-DAG: !rec_Last127 = !cir.struct<"Last127" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.int<s, 127, bitint>}>
// LLVM-DAG: %struct.Last127 = type <{ i32, [4 x i8], i128 }>

struct Last128 {
  int i;
  _BitInt(128) bi;
};
// CIR-DAG: !rec_Last128 = !cir.struct<"Last128" packed {data !s32i, pad !cir.array<!u8i x 4>, data !s128i_bitint}>
// LLVM-DAG: %struct.Last128 = type <{ i32, [4 x i8], i128 }>

struct First65 {
  _BitInt(65) bi;
  int i;
};
// CIR-DAG: !rec_First65 = !cir.struct<"First65" packed {data !cir.int<s, 65, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First65 = type <{ i128, i32, [4 x i8] }>

struct First127 {
  _BitInt(127) bi;
  int i;
};
// CIR-DAG: !rec_First127 = !cir.struct<"First127" packed {data !cir.int<s, 127, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First127 = type <{ i128, i32, [4 x i8] }>

struct First128 {
  _BitInt(128) bi;
  int i;
};
// CIR-DAG: !rec_First128 = !cir.struct<"First128" packed {data !s128i_bitint, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.First128 = type <{ i128, i32, [4 x i8] }>

struct Middle65 {
  char c;
  _BitInt(65) bi;
  int i;
};
// CIR-DAG: !rec_Middle65 = !cir.struct<"Middle65" packed {data !s8i, pad !cir.array<!u8i x 7>, data !cir.int<s, 65, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle65 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

struct Middle127 {
  char c;
  _BitInt(127) bi;
  int i;
};
// CIR-DAG: !rec_Middle127 = !cir.struct<"Middle127" packed {data !s8i, pad !cir.array<!u8i x 7>, data !cir.int<s, 127, bitint>, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle127 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

struct Middle128 {
  char c;
  _BitInt(128) bi;
  int i;
};
// CIR-DAG: !rec_Middle128 = !cir.struct<"Middle128" packed {data !s8i, pad !cir.array<!u8i x 7>, data !s128i_bitint, data !s32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.Middle128 = type <{ i8, [7 x i8], i128, i32, [4 x i8] }>

// Force emit.
struct Last65 l65[2];
struct Last127 l127[2];
struct Last128 l128[2];

struct First65 f65[2];
struct First127 f127[2];
struct First128 f128[2];

struct Middle65 m65[2];
struct Middle127 m127[2];
struct Middle128 m128[2];
