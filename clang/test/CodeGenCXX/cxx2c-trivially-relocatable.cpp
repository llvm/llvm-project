// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef __SIZE_TYPE__ size_t;

struct S trivially_relocatable_if_eligible {
    S(const S&);
    ~S();
    int a;
    int b;
};

// CHECK: @_Z4testP1SS0_
void test(S* source, S* dest, size_t count) {
    // CHECK: call void @llvm.memmove.p0.p0.i64({{.*}}, i64 8
    // CHECK-NOT: __builtin
    __builtin_trivially_relocate(dest, source, 1);
    // CHECK: [[A:%.*]] = load i64, ptr %count.addr
    // CHECK: [[M:%.*]] = mul i64 [[A]], 8
    // CHECK: call void @llvm.memmove.p0.p0.i64({{.*}}, i64 [[M]]
    __builtin_trivially_relocate(dest, source, count);
    // CHECK: ret
};
