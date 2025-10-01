// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc %s -emit-llvm \
// RUN:   -o - | FileCheck %s

template<typename T> struct S {};

// CHECK: void @"?f1@@YAXU?$S@U__SVInt8_t@__clang@@@@@Z"
void f1(S<__SVInt8_t>) {}
// CHECK: void @"?f2@@YAXU?$S@U__SVInt32_t@__clang@@@@@Z"
void f2(S<__SVInt32_t>) {}
// CHECK: void @"?f3@@YAXU?$S@U__SVBool_t@__clang@@@@@Z"
void f3(S<__SVBool_t>) {}
// CHECK: void @"?f4@@YAXU?$S@U__clang_svfloat64x4_t@__clang@@@@@Z"
void f4(S<__clang_svfloat64x4_t>) {}
