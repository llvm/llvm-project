// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Static local with trivial initialization.
//
// CodeGen:
//   @_ZZ10get_staticvE1x = internal global i32 42
//
// CIR:
//   Same, but check for any divergences in access

// DIFF: Check for static local handling

int get_static() {
    static int x = 42;
    return x++;
}

int test() {
    return get_static();
}
