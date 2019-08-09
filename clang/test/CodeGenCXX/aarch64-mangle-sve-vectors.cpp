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
