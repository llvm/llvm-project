// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct X { int a; };
struct Y { int b; };
struct Z : X, Y { int c; };

Z z = {{1}, {2}, 3};

// CIR: cir.global external @z = #cir.const_record<{#cir.const_record<{#cir.int<1> : !s32i}> : !rec_X, #cir.const_record<{#cir.int<2> : !s32i}> : !rec_Y, #cir.int<3> : !s32i}> : !rec_Z

// LLVM: @z = global %struct.Z { %struct.X { i32 1 }, %struct.Y { i32 2 }, i32 3 }

// OGCG: @z = global { i32, i32, i32 } { i32 1, i32 2, i32 3 }
