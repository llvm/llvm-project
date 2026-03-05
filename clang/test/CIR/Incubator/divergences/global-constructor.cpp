// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Global object with constructor.
//
// CodeGen:
//   @global_obj = global %struct.GlobalClass zeroinitializer
//   @llvm.global_ctors for initialization
//
// CIR:
//   Check for differences

// DIFF: Check for global constructor handling

struct GlobalClass {
    int value;
    GlobalClass(int v) : value(v) {}
};

GlobalClass global_obj(42);

int test() {
    return global_obj.value;
}
