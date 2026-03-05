// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Empty structs have size 1 in C++ (cannot be zero-sized).
// Per ABI, they may be ignored in parameter passing.
//
// CodeGen may omit or handle specially
// CIR may treat as regular struct

// DIFF: Check for empty struct handling

struct EmptyStruct {};

EmptyStruct return_empty() {
    return {};
}

void take_empty(EmptyStruct s) {}

int test() {
    EmptyStruct s = return_empty();
    take_empty(s);
    return 0;
}
