// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Mixed int and float struct has complex ABI classification.
//
// CodeGen handles mixed INTEGER/SSE classification:
//   (check specific lowering - may be split across registers)
//
// CIR may not handle correctly

// DIFF: Check for mixed int/float handling

struct MixedStruct {
    int i;      // INTEGER class
    float f;    // SSE class
    // Total 8 bytes
};

MixedStruct return_mixed() {
    return {42, 3.14f};
}

int test() {
    MixedStruct s = return_mixed();
    return s.i;
}
