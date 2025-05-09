// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct S trivially_relocatable_if_eligible {
    S(const S&);
    ~S();
    int a;
    int b;
};

// CHECK: @_Z4testP1SS0_
// CHECK: call void @llvm.memmove.p0.p0.i64
// CHECK-NOT: __builtin
// CHECK: ret
void test(S* source, S* dest) {
    __builtin_trivially_relocate(dest, source, 1);
};
