// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=A64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=A64-LLVM,A64-LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=A64-LLVM,A64-OGCG --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=X64-CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexperimental-max-bitint-width=256 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=X64-LLVM --input-file=%t.ll %s

// The _BitInt alignment cap is TargetInfo::getBitIntMaxAlign(). AArch64 sets
// it to 128 bits, so a _BitInt wider than 64 bits is 16-byte aligned and
// _BitInt(129) is stored as i256. x86_64 leaves the cap at LongLongAlign
// (64 bits), so the same values are 8-byte aligned and _BitInt(129) is 24
// bytes.

signed _BitInt(65) g65 = 1;
signed _BitInt(128) g128 = 1;
signed _BitInt(129) g129 = -1;

// A64-CIR-DAG: !cir.int<u, 1, bitint> = #cir.bitint_spec<max_align = 128>
// A64-CIR-DAG: cir.global external @g65 = #cir.int<1> : !cir.int<s, 65, bitint> align(16)
// A64-CIR-DAG: cir.global external @g128 = #cir.int<1> : !s128i_bitint align(16)
// A64-CIR-DAG: cir.global external @g129 = #cir.int<-1> : !cir.int<s, 129, bitint> align(16)
// A64-LLVM-DAG: @g65 = global i128 1, align 16
// A64-LLVM-DAG: @g128 = global i128 1, align 16
// A64-LLVM-DAG: @g129 = global i256 -1, align 16

// X64-CIR-DAG: !cir.int<u, 1, bitint> = #cir.bitint_spec<max_align = 64>
// X64-CIR-DAG: cir.global external @g65 = #cir.int<1> : !cir.int<s, 65, bitint> align(8)
// X64-CIR-DAG: cir.global external @g128 = #cir.int<1> : !s128i_bitint align(8)
// X64-CIR-DAG: cir.global external @g129 = #cir.int<-1> : !cir.int<s, 129, bitint> align(8)
// X64-LLVM-DAG: @g65 = global i128 1, align 8
// X64-LLVM-DAG: @g128 = global i128 1, align 8
// X64-LLVM-DAG: @g129 = global [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF", align 8

struct Has129 {
  int i;
  signed _BitInt(129) bi;
};
struct Has129 g_struct = {7, -1};

// AArch64: the i256 member is 16-byte aligned. CIR leaves the 12-byte gap
// to LLVM struct layout. Classic codegen writes the padding as an array.
// A64-CIR-DAG: !rec_Has129 = !cir.struct<"Has129" {data !s32i, data !cir.int<s, 129, bitint>}>
// A64-CIR-DAG: cir.global external @g_struct = #cir.const_record<{#cir.int<7> : !s32i, #cir.int<-1> : !cir.int<s, 129, bitint>}> : !rec_Has129 align(16)
// A64-LLVMCIR-DAG: @g_struct = global %struct.Has129 { i32 7, i256 -1 }, align 16
// A64-OGCG-DAG: @g_struct = global { i32, [12 x i8], i256 } { i32 7, [12 x i8] zeroinitializer, i256 -1 }, align 16

// x86_64: 4 bytes of padding bring the 24-byte _BitInt to its 8-byte
// alignment. CIR packs the struct. Classic codegen does not.
// X64-CIR-DAG: !rec_Has129 = !cir.struct<"Has129" packed {data !s32i, pad !cir.array<!u8i x 4>, data !cir.int<s, 129, bitint>}>
// X64-CIR-DAG: cir.global external @g_struct = #cir.const_record<{#cir.int<7> : !s32i, #cir.zero : !cir.array<!u8i x 4>, #cir.int<-1> : !cir.int<s, 129, bitint>}> : !rec_Has129 align(8)
// X64-LLVM-DAG: %struct.Has129 = type {{[<]?}}{ i32, [4 x i8], [24 x i8] }{{[>]?}}
// X64-LLVM-DAG: @g_struct = global %struct.Has129 {{[<]?}}{ i32 7, [4 x i8] zeroinitializer, [24 x i8] c"\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF\FF" }{{[>]?}}, align 8
