// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Diamond inheritance (non-virtual) missing comdat.
//
// CodeGen:
//   $_ZN7DiamondC1Ev = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_ZN7DiamondC1Ev = comdat any

struct Base {
    int x = 1;
};

struct Left : Base {
    int y = 2;
};

struct Right : Base {
    int z = 3;
};

struct Diamond : Left, Right {
    int w = 4;
};

int test() {
    Diamond d;
    return d.Left::x + d.Right::x + d.y + d.z + d.w;
}
