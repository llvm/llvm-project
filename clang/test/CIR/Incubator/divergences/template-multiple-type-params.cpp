// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Template with multiple type parameters missing comdat.
//
// CodeGen:
//   $_Z7processIiiEvT_T0_ = comdat any
//
// CIR:
//   Missing comdat

// DIFF: -$_Z7processIiiEvT_T0_ = comdat any
// DIFF: +# Missing comdat

template<typename T, typename U>
void process(T t, U u) {}

int test() {
    process(10, 20);
    return 0;
}
