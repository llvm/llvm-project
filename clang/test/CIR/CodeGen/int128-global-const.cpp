// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

unsigned __int128 a = ((unsigned __int128)1 << 70) + 3;
__int128 b = -((__int128)1 << 70);

// CIR: cir.global external @a = #cir.int<1180591620717411303427> : !u128i
// CIR: cir.global external @b = #cir.int<-1180591620717411303424> : !s128i

// LLVM: @a = global i128 1180591620717411303427
// LLVM: @b = global i128 -1180591620717411303424
