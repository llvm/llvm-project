// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: diff -u %t.ll %t.cir.ll | FileCheck %s --check-prefix=DIFF
//
// XFAIL: *
//
// Pointer to overloaded member function requires explicit type cast.
//
// CodeGen decomposes:
//   define i32 @call_overloaded(ptr %s, i64 %ptr.coerce0, i64 %ptr.coerce1)
//
// CIR passes as struct:
//   define i32 @call_overloaded(ptr %0, { i64, i64 } %1)

// DIFF: -define {{.*}} @{{.*}}call_overloaded(ptr{{.*}}, i64{{.*}}, i64
// DIFF: +define {{.*}} @{{.*}}call_overloaded(ptr{{.*}}, { i64, i64 }

struct S {
    int foo(int x) { return x; }
    int foo(double x) { return static_cast<int>(x); }
};

int call_overloaded(S* s, int (S::*ptr)(int)) {
    return (s->*ptr)(42);
}

int test() {
    S s;
    return call_overloaded(&s, static_cast<int (S::*)(int)>(&S::foo));
}
