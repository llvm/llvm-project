// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,OGCG

struct B1 { unsigned int a : 3; unsigned int b : 5; };
struct B1 b1 = { 0b101, 0b01111 };
// CIR: cir.global external @b1 = #cir.const_record<{#cir.int<125> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_B1
// LLVM: @b1 = global %struct.B1 { i8 125, [3 x i8] zeroinitializer }

struct B2 { unsigned int a : 8; unsigned int b : 8; unsigned int c : 16; };
struct B2 b2 = { 0xAA, 0xBB, 0xCCDD };
// CIR: cir.global external @b2 = #cir.const_record<{#cir.int<3437083562> : !u32i}> : !rec_B2
// LLVMCIR: @b2 = global %struct.B2 { i32 -857883734 }
// OGCG: @b2 = global { i8, i8, i8, i8 } { i8 -86, i8 -69, i8 -35, i8 -52 }

struct BP { unsigned int a : 3; unsigned int b : 5; int c; };
struct BP bp = { 1, 2, 99 };
// CIR: cir.global external @bp = #cir.const_record<{#cir.int<17> : !u8i, #cir.int<99> : !s32i}> : !rec_BP
// LLVMCIR: @bp = global %struct.BP { i8 17, i32 99 }
// OGCG: @bp = global { i8, [3 x i8], i32 } { i8 17, [3 x i8] zeroinitializer, i32 99 }

struct BS { int a : 4; int b : 4; };
struct BS bs = { -1, 3 };
// CIR: cir.global external @bs = #cir.const_record<{#cir.int<63> : !u8i, #cir.zero : !cir.array<!u8i x 3>}> : !rec_BS
// LLVM: @bs = global %struct.BS { i8 63, [3 x i8] zeroinitializer }
