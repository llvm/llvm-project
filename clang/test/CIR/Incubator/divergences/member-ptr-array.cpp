// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Array of member pointers.
// Each element is {i64, i64} but array indexing and access still needs proper ABI.
//
// CodeGen handles array properly
// CIR may have issues

// DIFF: Check for member pointer array handling

struct S {
    int a, b, c;
};

int access_by_index(S* s, int index) {
    int S::*ptrs[] = {&S::a, &S::b, &S::c};
    return s->*ptrs[index];
}

int test() {
    S s{10, 20, 30};
    return access_by_index(&s, 1);
}
