// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Member pointer to base class member accessed via derived.
//
// CodeGen decomposes:
//   define i32 @access_base_member(ptr %d, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @access_base_member(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}access_base_member(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}access_base_member(ptr{{.*}}, { i64, i64 }

struct Base {
    int x;
};

struct Derived : Base {
    int y;
};

int access_base_member(Derived* d, int Base::*ptr) {
    return d->*ptr;
}

int test() {
    Derived d{{42}, 100};
    return access_base_member(&d, &Base::x);
}
