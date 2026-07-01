// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

char high_bytes[] = "\x80\xff\x7f";

// CIR: cir.global external @high_bytes = #cir.const_array<"\80\FF\7F" : !cir.array<!s8i x 3>, trailing_zeros> : !cir.array<!s8i x 4>
// LLVM: @high_bytes = global [4 x i8] c"\80\FF\7F\00"

unsigned char ubytes[4] = {0x80, 0xff, 0x01, 0x7f};

// CIR: cir.global external @ubytes = #cir.const_array<[#cir.int<128> : !u8i, #cir.int<255> : !u8i, #cir.int<1> : !u8i, #cir.int<127> : !u8i]> : !cir.array<!u8i x 4>
// LLVM: @ubytes = global [4 x i8] c"\80\FF\01\7F"
