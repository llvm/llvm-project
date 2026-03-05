// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Member pointer comparison.
//
// CodeGen decomposes both member pointers:
//   define i1 @compare_member_ptrs(i64 %p1.coerce0, i64 %p1.coerce1, i64 %p2.coerce0, i64 %p2.coerce1)
//
// CIR passes as structs:
//   define i1 @compare_member_ptrs({ i64, i64 } %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}compare_member_ptrs(i64{{.*}}, i64{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}compare_member_ptrs({ i64, i64 }{{.*}}, { i64, i64 }

struct S {
    int x, y;
};

bool compare_member_ptrs(int S::*p1, int S::*p2) {
    return p1 == p2;
}

int test() {
    return compare_member_ptrs(&S::x, &S::x) ? 1 : 0;
}
