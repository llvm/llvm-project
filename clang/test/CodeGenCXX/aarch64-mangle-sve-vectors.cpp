// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu %s -emit-llvm -o - \
// RUN:   -target-feature +sve | FileCheck %s

template<typename T> struct S {};

// CHECK: _Z2f11SIu10__SVInt8_tE
void f1(S<__SVInt8_t>) {}
// CHECK: _Z2f21SIu11__SVInt16_tE
void f2(S<__SVInt16_t>) {}
// CHECK: _Z2f31SIu11__SVInt32_tE
void f3(S<__SVInt32_t>) {}
// CHECK: _Z2f41SIu11__SVInt64_tE
void f4(S<__SVInt64_t>) {}
// CHECK: _Z2f51SIu11__SVUint8_tE
void f5(S<__SVUint8_t>) {}
// CHECK: _Z2f61SIu12__SVUint16_tE
void f6(S<__SVUint16_t>) {}
// CHECK: _Z2f71SIu12__SVUint32_tE
void f7(S<__SVUint32_t>) {}
// CHECK: _Z2f81SIu12__SVUint64_tE
void f8(S<__SVUint64_t>) {}
// CHECK: _Z2f91SIu13__SVFloat16_tE
void f9(S<__SVFloat16_t>) {}
// CHECK: _Z3f101SIu13__SVFloat32_tE
void f10(S<__SVFloat32_t>) {}
// CHECK: _Z3f111SIu13__SVFloat64_tE
void f11(S<__SVFloat64_t>) {}
// CHECK: _Z3f121SIu10__SVBool_tE
void f12(S<__SVBool_t>) {}

// The tuple types don't use the internal name for mangling.

// CHECK: _Z3f131SI10svint8x2_tE
void f13(S<__clang_svint8x2_t>) {}
// CHECK: _Z3f141SI10svint8x3_tE
void f14(S<__clang_svint8x3_t>) {}
// CHECK: _Z3f151SI10svint8x4_tE
void f15(S<__clang_svint8x4_t>) {}
// CHECK: _Z3f161SI11svint16x2_tE
void f16(S<__clang_svint16x2_t>) {}
// CHECK: _Z3f171SI11svint16x3_tE
void f17(S<__clang_svint16x3_t>) {}
// CHECK: _Z3f181SI11svint16x4_tE
void f18(S<__clang_svint16x4_t>) {}
// CHECK: _Z3f191SI11svint32x2_tE
void f19(S<__clang_svint32x2_t>) {}
// CHECK: _Z3f201SI11svint32x3_tE
void f20(S<__clang_svint32x3_t>) {}
// CHECK: _Z3f211SI11svint32x4_tE
void f21(S<__clang_svint32x4_t>) {}
// CHECK: _Z3f221SI11svint64x2_tE
void f22(S<__clang_svint64x2_t>) {}
// CHECK: _Z3f231SI11svint64x3_tE
void f23(S<__clang_svint64x3_t>) {}
// CHECK: _Z3f241SI11svint64x4_tE
void f24(S<__clang_svint64x4_t>) {}
// CHECK: _Z3f251SI11svuint8x2_tE
void f25(S<__clang_svuint8x2_t>) {}
// CHECK: _Z3f261SI11svuint8x3_tE
void f26(S<__clang_svuint8x3_t>) {}
// CHECK: _Z3f271SI11svuint8x4_tE
void f27(S<__clang_svuint8x4_t>) {}
// CHECK: _Z3f281SI12svuint16x2_tE
void f28(S<__clang_svuint16x2_t>) {}
// CHECK: _Z3f291SI12svuint16x3_tE
void f29(S<__clang_svuint16x3_t>) {}
// CHECK: _Z3f301SI12svuint16x4_tE
void f30(S<__clang_svuint16x4_t>) {}
// CHECK: _Z3f311SI12svuint32x2_tE
void f31(S<__clang_svuint32x2_t>) {}
// CHECK: _Z3f321SI12svuint32x3_tE
void f32(S<__clang_svuint32x3_t>) {}
// CHECK: _Z3f331SI12svuint32x4_tE
void f33(S<__clang_svuint32x4_t>) {}
// CHECK: _Z3f341SI12svuint64x2_tE
void f34(S<__clang_svuint64x2_t>) {}
// CHECK: _Z3f351SI12svuint64x3_tE
void f35(S<__clang_svuint64x3_t>) {}
// CHECK: _Z3f361SI12svuint64x4_tE
void f36(S<__clang_svuint64x4_t>) {}
// CHECK: _Z3f371SI13svfloat16x2_tE
void f37(S<__clang_svfloat16x2_t>) {}
// CHECK: _Z3f381SI13svfloat16x3_tE
void f38(S<__clang_svfloat16x3_t>) {}
// CHECK: _Z3f391SI13svfloat16x4_tE
void f39(S<__clang_svfloat16x4_t>) {}
// CHECK: _Z3f401SI13svfloat32x2_tE
void f40(S<__clang_svfloat32x2_t>) {}
// CHECK: _Z3f411SI13svfloat32x3_tE
void f41(S<__clang_svfloat32x3_t>) {}
// CHECK: _Z3f421SI13svfloat32x4_tE
void f42(S<__clang_svfloat32x4_t>) {}
// CHECK: _Z3f431SI13svfloat64x2_tE
void f43(S<__clang_svfloat64x2_t>) {}
// CHECK: _Z3f441SI13svfloat64x3_tE
void f44(S<__clang_svfloat64x3_t>) {}
// CHECK: _Z3f451SI13svfloat64x4_tE
void f45(S<__clang_svfloat64x4_t>) {}
