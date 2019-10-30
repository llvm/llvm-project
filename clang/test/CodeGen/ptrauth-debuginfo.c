// RUN: %clang_cc1 -triple arm64-apple-ios \
// RUN:   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -fblocks \
// RUN:   %s -debug-info-kind=limited -o - | FileCheck %s

// Constant initializers for data pointers.
extern int external_int;

// CHECK: !DIDerivedType(tag: DW_TAG_APPLE_ptrauth_type,
// CHECK-SAME:           ptrAuthKey: 1,
// CHECK-SAME:           ptrAuthIsAddressDiscriminated: false,
// CHECK-SAME:           ptrAuthExtraDiscriminator: 1234)
int * __ptrauth(1,0,1234) g1 = &external_int;

struct A {
  int value;
};
struct A *createA(void);

void f() {
  __block struct A * __ptrauth(1, 1, 1) ptr = createA();
  ^{ ptr->value; }();
}
// CHECK: !DIDerivedType(tag: DW_TAG_APPLE_ptrauth_type,
// CHECK-SAME:           ptrAuthKey: 1,
// CHECK-SAME:           ptrAuthIsAddressDiscriminated: true,
// CHECK-SAME:           ptrAuthExtraDiscriminator: 1)
