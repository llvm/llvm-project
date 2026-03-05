// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Struct with single float should be passed in XMM register per x86_64 ABI.
//
// CodeGen handles SSE classification:
//   (check specific lowering)
//
// CIR may not handle correctly

// DIFF: Check for float struct handling

struct FloatStruct {
    float f;  // 4 bytes, SSE class
};

FloatStruct return_float() {
    return {3.14f};
}

int test() {
    FloatStruct s = return_float();
    return s.f > 3.0f ? 1 : 0;
}
