// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Structs with bitfields have special layout and ABI classification.
//
// CodeGen handles bitfield packing in calling convention
// CIR may have different bitfield struct handling

// DIFF: Check for bitfield struct differences

struct BitfieldStruct {
    unsigned int a : 3;  // 3 bits
    unsigned int b : 5;  // 5 bits
    // Total: 1 byte with padding
};

BitfieldStruct return_bitfield() {
    return {1, 2};
}

int test() {
    BitfieldStruct s = return_bitfield();
    return s.a + s.b;
}
