// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Pointer to member function with multiple inheritance.
// Member function pointers need offset adjustment for non-first base.
//
// CodeGen decomposes {i64, i64}:
//   define i32 @access_b(ptr %c, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @access_b(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}access_b(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}access_b(ptr{{.*}}, { i64, i64 }

struct A { int a; };
struct B { int b; };
struct C : A, B { int c; };

int access_b(C* c, int B::*ptr) {
    return c->*ptr;
}

int test() {
    C c{{1}, {2}, 3};
    return access_b(&c, &B::b);
}
