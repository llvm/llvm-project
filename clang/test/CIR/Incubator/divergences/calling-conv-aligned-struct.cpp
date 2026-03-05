// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Over-aligned structs (alignas > natural alignment) affect ABI classification.
// This struct has only 4 bytes but requires 32-byte alignment.
//
// CodeGen handles alignment in calling convention
// CIR may not properly handle over-aligned struct returns

// DIFF: Check for alignment handling differences

struct alignas(32) AlignedStruct {
    int x;  // 4 bytes but 32-byte aligned
};

AlignedStruct return_aligned() {
    return {42};
}

int test() {
    AlignedStruct s = return_aligned();
    return s.x;
}
