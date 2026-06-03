// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Pin down the packed value of bitfield-bearing globals across the whole
// emission path. The CIR record layout collapses contiguous bitfields into a
// single storage member, and the constant initializer must agree on both the
// packed integer value and the placement of any non-bitfield fields.

// Two unsigned bitfields packed into one byte: a=0b101 in bits[0..2], b=0b01111
// in bits[3..7]. Packed value = 0b01111101 = 125.
struct B1 { unsigned int a : 3; unsigned int b : 5; };
struct B1 b1 = { 0b101, 0b01111 };

// CIR: cir.global external @b1 = #cir.const_record<{#cir.int<125> : !u8i,
// LLVM: @b1 ={{.*}}i8 125
// OGCG: @b1 ={{.*}}i8 125

// Byte-aligned bitfields packed into a single 32-bit storage word: a=0xAA in
// bits[0..7], b=0xBB in bits[8..15], c=0xCCDD in bits[16..31]. Packed value
// (little-endian) is 0xCCDDBBAA = -857883734 as i32.
struct B2 { unsigned int a : 8; unsigned int b : 8; unsigned int c : 16; };
struct B2 b2 = { 0xAA, 0xBB, 0xCCDD };

// LLVM: @b2 ={{.*}}i32 -857883734
// OGCG: @b2 ={{.*}}i8 -86, i8 -69, i8 -35, i8 -52

// Bitfield storage followed by an alignment gap and a non-bitfield field.
// a=1 (bits[0..2]), b=2 (bits[3..7]) -> packed = 0b00010001 = 17.
// Then c=99 at byte offset 4.
struct BP { unsigned int a : 3; unsigned int b : 5; int c; };
struct BP bp = { 1, 2, 99 };

// LLVM: @bp ={{.*}}i8 17,{{.*}}i32 99
// OGCG: @bp ={{.*}}i8 17,{{.*}}i32 99

// Signed bitfields. a = -1 (4 bits) = 0b1111; b = 3 (4 bits) = 0b0011.
// Packed little-endian into one byte: 0b00111111 = 63.
struct BS { int a : 4; int b : 4; };
struct BS bs = { -1, 3 };

// LLVM: @bs ={{.*}}i8 63
// OGCG: @bs ={{.*}}i8 63
