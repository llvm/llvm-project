// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Pointer to const member function.
//
// CodeGen decomposes:
//   define i32 @call_const_method(ptr %s, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @call_const_method(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}call_const_method(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}call_const_method(ptr{{.*}}, { i64, i64 }

struct S {
    int get() const { return 42; }
};

int call_const_method(const S* s, int (S::*ptr)() const) {
    return (s->*ptr)();
}

int test() {
    S s;
    return call_const_method(&s, &S::get);
}
