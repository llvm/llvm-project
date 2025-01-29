// REQUIRES: arm-registered-target

// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=SIGN
// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=BTE
// RUN: %clang_cc1 -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=ALL

// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=SIGN
// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=BTE
// RUN: %clang_cc1 -flto -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=ALL

// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -msign-return-address=non-leaf %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=SIGN
// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce  %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=BTE
// RUN: %clang_cc1 -flto=thin -triple=thumbv7m-unknown-unknown-eabi -mbranch-target-enforce -msign-return-address=all %s -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=ALL

void foo() {}

// Check there are branch protection function attributes.
// CHECK-LABEL: @foo() #[[#ATTR:]]

// SIGN-NOT: attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"
// SIGN: attributes #[[#ATTR]] = { {{.*}} "sign-return-address"="non-leaf"
// BTE-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address"
// BTE:  attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"
// ALL:  attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"{{.*}} "sign-return-address"="all"
