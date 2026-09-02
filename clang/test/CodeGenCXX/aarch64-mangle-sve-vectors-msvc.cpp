// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc \
// RUN:   -target-feature +sve -emit-llvm -o - %s | FileCheck %s

template <typename T> struct S {};

// Scalar SVE types.

// CHECK: void @"?f1@@YAXU?$S@$_CB@@@Z"
void f1(S<__SVInt8_t>) {}

// CHECK: void @"?f2@@YAXU?$S@$_CD@@@Z"
void f2(S<__SVInt32_t>) {}

// CHECK: void @"?f3@@YAXU?$S@$_CA@@@Z"
void f3(S<__SVBool_t>) {}

// CHECK: void @"?f4@@YAXU?$S@$_CH@@@Z"
void f4(S<__SVUint32_t>) {}

// CHECK: void @"?f5@@YAXU?$S@$_CL@@@Z"
void f5(S<__SVFloat32_t>) {}

// CHECK: void @"?f13@@YAXU?$S@$_CJ@@@Z"
void f13(S<__SVBfloat16_t>) {}

// Tuple SVE types.

// CHECK: void @"?f6@@YAXU?$S@$_C2B@@@Z"
void f6(S<__clang_svint8x2_t>) {}

// CHECK: void @"?f7@@YAXU?$S@$_C3B@@@Z"
void f7(S<__clang_svint8x3_t>) {}

// CHECK: void @"?f8@@YAXU?$S@$_C4B@@@Z"
void f8(S<__clang_svint8x4_t>) {}

// CHECK: void @"?f9@@YAXU?$S@$_C4M@@@Z"
void f9(S<__clang_svfloat64x4_t>) {}

// CHECK: void @"?f14@@YAXU?$S@$_C2H@@@Z"
void f14(S<__clang_svuint32x2_t>) {}

// CHECK: void @"?f15@@YAXU?$S@$_C3L@@@Z"
void f15(S<__clang_svfloat32x3_t>) {}

// Unsupported types should continue using legacy artificial tag mangling.

// CHECK: void @"?f10@@YAXU?$S@U__SVMfloat8_t@__clang@@@@@Z"
void f10(S<__SVMfloat8_t>) {}

// CHECK: void @"?f11@@YAXU?$S@U__clang_svboolx2_t@__clang@@@@@Z"
void f11(S<__clang_svboolx2_t>) {}

// CHECK: void @"?f12@@YAXU?$S@U__SVCount_t@__clang@@@@@Z"
void f12(S<__SVCount_t>) {}
