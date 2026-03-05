// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Struct with two floats (8 bytes total) fits in one XMM register.
//
// CodeGen handles SSE classification:
//   (check specific lowering)
//
// CIR may not handle correctly

// DIFF: Check for two floats struct handling

struct TwoFloats {
    float a, b;  // 8 bytes, SSE class
};

TwoFloats return_two_floats() {
    return {1.0f, 2.0f};
}

int test() {
    TwoFloats s = return_two_floats();
    return s.a + s.b > 2.5f ? 1 : 0;
}
