// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIRLE

// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIRBE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVMLE,LLVMCIRLE
//
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVMBE,LLVMCIRBE
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVMLE,OGCGLE
//
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVMBE,OGCGBE

struct B1 { unsigned int a : 3; unsigned int b : 5; };
struct B1 b1 = { 0b101, 0b01111 };
// CIRLE: cir.global external @b1 = #cir.const_record<{#cir.int<125> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_B1
// CIRBE: cir.global external @b1 = #cir.const_record<{#cir.int<175> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_B1
// LLVMLE: @b1 = global %struct.B1 { i8 125, [3 x i8] zeroinitializer }
// LLVMBE: @b1 = global %struct.B1 { i8 -81, [3 x i8] zeroinitializer }

struct B2 { unsigned int a : 8; unsigned int b : 8; unsigned int c : 16; };
struct B2 b2 = { 0xAA, 0xBB, 0xCCDD };
// CIRLE: cir.global external @b2 = #cir.const_record<{#cir.int<3437083562> : !u32i}> : !rec_B2
// CIRBE: cir.global external @b2 = #cir.const_record<{#cir.int<2864434397> : !u32i}> : !rec_B2
// LLVMCIRLE: @b2 = global %struct.B2 { i32 -857883734 }
// LLVMCIRBE: @b2 = global %struct.B2 { i32 -1430532899 }
// OGCGLE: @b2 = global { i8, i8, i8, i8 } { i8 -86, i8 -69, i8 -35, i8 -52 }
// OGCGBE: @b2 = global { i8, i8, i8, i8 } { i8 -86, i8 -69, i8 -52, i8 -35 }

struct BP { unsigned int a : 3; unsigned int b : 5; int c; };
struct BP bp = { 1, 2, 99 };
// CIRLE: cir.global external @bp = #cir.const_record<{#cir.int<17> : !u8i, #cir.int<99> : !s32i}> : !rec_BP
// CIRBE: cir.global external @bp = #cir.const_record<{#cir.int<34> : !u8i, #cir.int<99> : !s32i}> : !rec_BP
// LLVMCIRLE: @bp = global %struct.BP { i8 17, i32 99 }
// LLVMCIRBE: @bp = global %struct.BP { i8 34, i32 99 }
// OGCGLE: @bp = global { i8, [3 x i8], i32 } { i8 17, [3 x i8] zeroinitializer, i32 99 }
// OGCGBE: @bp = global { i8, [3 x i8], i32 } { i8 34, [3 x i8] zeroinitializer, i32 99 }

struct BS { int a : 4; int b : 4; };
struct BS bs = { -1, 3 };
// CIRLE: cir.global external @bs = #cir.const_record<{#cir.int<63> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_BS
// CIRBE: cir.global external @bs = #cir.const_record<{#cir.int<243> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_BS
// LLVMLE: @bs = global %struct.BS { i8 63, [3 x i8] zeroinitializer }
// LLVMBE: @bs = global %struct.BS { i8 -13, [3 x i8] zeroinitializer }

struct BA { int a : 24; char c; };
struct BA ba = { 0x123456, 7 };
// CIRLE:  cir.global external @ba = #cir.const_record<{#cir.const_array<[#cir.int<86> : !u8i, #cir.int<52> : !u8i, #cir.int<18> : !u8i]> : !cir.array<!u8i x 3>, #cir.int<7> : !s8i}> : !rec_BA
// CIRBE:  cir.global external @ba = #cir.const_record<{#cir.const_array<[#cir.int<18> : !u8i, #cir.int<52> : !u8i, #cir.int<86> : !u8i]> : !cir.array<!u8i x 3>, #cir.int<7> : !s8i}> : !rec_BA
// LLVMCIRLE: @ba = global %struct.BA { [3 x i8] c"V4\12", i8 7 }
// LLVMCIRBE: @ba = global %struct.BA { [3 x i8] c"\124V", i8 7 }
// OGCGLE: @ba = global { i8, i8, i8, i8 } { i8 86, i8 52, i8 18, i8 7 }
// OGCGBE: @ba = global { i8, i8, i8, i8 } { i8 18, i8 52, i8 86, i8 7 }
