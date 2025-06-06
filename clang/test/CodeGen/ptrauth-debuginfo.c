// RUN: %clang_cc1 -triple arm64-apple-ios \
// RUN:   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -fblocks \
// RUN:   %s -debug-info-kind=limited -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu \
// RUN:   -fptrauth-calls -fptrauth-intrinsics -emit-llvm -fblocks \
// RUN:   %s -debug-info-kind=limited -o - | FileCheck %s

// Constant initializers for data pointers.
extern int external_int;

int *__ptrauth(1, 0, 1234) g1 = &external_int;
// CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type,
// CHECK-SAME:           ptrAuthKey: 1,
// CHECK-SAME:           ptrAuthIsAddressDiscriminated: false,
// CHECK-SAME:           ptrAuthExtraDiscriminator: 1234,
// CHECK-SAME:           ptrAuthIsaPointer: false,
// CHECK-SAME:           ptrAuthAuthenticatesNullValues: false)

struct A {
  int value;
};
struct A *createA(void);

void f() {
  __block struct A *__ptrauth(0, 1, 1236) ptr = createA();
  ^{
    (void)ptr->value;
  }();
}
// CHECK: !DIDerivedType(tag: DW_TAG_LLVM_ptrauth_type,
// CHECK-NOT:            ptrAuthKey
// CHECK-SAME:           ptrAuthIsAddressDiscriminated: true,
// CHECK-SAME:           ptrAuthExtraDiscriminator: 1236,
// CHECK-SAME:           ptrAuthIsaPointer: false,
// CHECK-SAME:           ptrAuthAuthenticatesNullValues: false)
