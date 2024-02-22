// RUN: %clang_cc1 %s                                                      -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-NONE
// RUN: %clang_cc1 %s                         -fptrauth-kernel-abi-version -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-WITH --check-prefix=CHECK-ZEROK
// RUN: %clang_cc1 %s -fptrauth-abi-version=0                              -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-WITH --check-prefix=CHECK-ZERO
// RUN: %clang_cc1 %s -fptrauth-abi-version=0 -fptrauth-kernel-abi-version -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-WITH --check-prefix=CHECK-ZEROK
// RUN: %clang_cc1 %s -fptrauth-abi-version=5                              -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-WITH --check-prefix=CHECK-FIVE
// RUN: %clang_cc1 %s -fptrauth-abi-version=5 -fptrauth-kernel-abi-version -triple arm64e-apple-ios -disable-llvm-passes -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-WITH --check-prefix=CHECK-FIVEK

int f(void) {
  return 0;
}
// CHECK-NONE-NOT: ptrauth.abi-version
// CHECK-WITH:  !llvm.module.flags = !{{{.*}} ![[ABI_VERSION_REF:[0-9]+]]}
// CHECK-WITH:  ![[ABI_VERSION_REF]] = !{i32 6, !"ptrauth.abi-version", ![[ABI_VERSION_VAR:[0-9]+]]}
// CHECK-WITH:  ![[ABI_VERSION_VAR]] = !{![[ABI_VERSION_VAL:[0-9]+]]}
// CHECK-ZERO:  ![[ABI_VERSION_VAL]] = !{i32 0, i1 false}
// CHECK-ZEROK: ![[ABI_VERSION_VAL]] = !{i32 0, i1 true}
// CHECK-FIVE:  ![[ABI_VERSION_VAL]] = !{i32 5, i1 false}
// CHECK-FIVEK: ![[ABI_VERSION_VAL]] = !{i32 5, i1 true}
