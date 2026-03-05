// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Pointer to data member calling convention.
// Member pointers are {i64, i64} and should be decomposed per x86_64 ABI.
//
// CodeGen decomposes:
//   define i32 @access_member(ptr %s, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @access_member(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}access_member(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}access_member(ptr{{.*}}, { i64, i64 }

struct S {
    int x, y;
};

int access_member(S* s, int S::*ptr) {
    return s->*ptr;
}

int test() {
    S s{1, 2};
    return access_member(&s, &S::x);
}
