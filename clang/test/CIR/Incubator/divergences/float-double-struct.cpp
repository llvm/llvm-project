// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Struct with double should be passed in XMM register per x86_64 ABI.
//
// CodeGen handles SSE classification:
//   (check specific lowering)
//
// CIR may not handle correctly

// DIFF: Check for double struct handling

struct DoubleStruct {
    double d;  // 8 bytes, SSE class
};

DoubleStruct return_double() {
    return {3.14159};
}

int test() {
    DoubleStruct s = return_double();
    return s.d > 3.0 ? 1 : 0;
}
