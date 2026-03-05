// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Null member pointer should be {0, 0}.
//
// CodeGen decomposes:
//   define i1 @test_null(i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i1 @test_null({ i64, i64 } %0)

// DIFF: -define {{.*}} @{{.*}}test_null(i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}test_null({ i64, i64 }

struct S {
    int x;
};

bool test_null(int S::*ptr) {
    return ptr == nullptr;
}

int test() {
    return test_null(nullptr) ? 1 : 0;
}
