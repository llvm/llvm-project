// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s --check-prefix=OGCG
//
// XFAIL: *
//
// CIR uses incorrect calling convention for member function pointers.
//
// Member function pointers are represented as { i64, i64 } structs containing:
// - Function pointer or vtable offset (first i64)
// - This-pointer adjustment (second i64)
//
// Per the System V x86_64 ABI, small structs should be passed in registers
// by decomposing them into separate scalar arguments.
//
// Current divergence:
// CIR: define i32 @_Z15call_member_ptrP1SMS_FiiEi(ptr %0, { i64, i64 } %1, i32 %2)
//      (passes struct directly)
//
// CodeGen: define i32 @_Z15call_member_ptrP1SMS_FiiEi(ptr %s, i64 %ptr.coerce0, i64 %ptr.coerce1, i32 %val)
//          (decomposes struct into two i64 parameters)
//
// This breaks ABI compatibility when calling functions across TUs.

struct S {
    int x;
    int foo(int y) { return x + y; }
};

int call_member_ptr(S* s, int (S::*ptr)(int), int val) {
    return (s->*ptr)(val);
}

int test() {
    S s;
    s.x = 42;
    return call_member_ptr(&s, &S::foo, 10);
}

// LLVM: Should decompose member pointer struct
// LLVM: define {{.*}} i32 @_Z15call_member_ptrP1SMS_FiiEi(ptr {{.*}}, { i64, i64 } {{.*}}, i32 {{.*}})

// OGCG: Should pass member pointer as two scalars
// OGCG: define {{.*}} i32 @_Z15call_member_ptrP1SMS_FiiEi(ptr {{.*}} %s, i64 %ptr.coerce0, i64 %ptr.coerce1, i32 {{.*}} %val)
