// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=none     %s | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=all      %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -msign-return-address=non-leaf %s | FileCheck %s --check-prefix=CHECK --check-prefix=PART

// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=none %s          | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+leaf  %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key %s | FileCheck %s --check-prefix=CHECK --check-prefix=B-KEY
// RUN: %clang -target aarch64-arm-none-eabi -S -emit-llvm -o - -mbranch-protection=bti %s           | FileCheck %s --check-prefix=CHECK --check-prefix=BTE

// REQUIRES: aarch64-registered-target

// Check there are no branch protection function attributes

// CHECK-LABEL: @foo() #[[#ATTR:]]

// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"

// Check module attributes

// NONE:  !{i32 8, !"branch-target-enforcement", i32 0}
// ALL:   !{i32 8, !"branch-target-enforcement", i32 0}
// PART:  !{i32 8, !"branch-target-enforcement", i32 0}
// BTE:   !{i32 8, !"branch-target-enforcement", i32 1}
// B-KEY: !{i32 8, !"branch-target-enforcement", i32 0}

// NONE:  !{i32 8, !"sign-return-address", i32 0}
// ALL:   !{i32 8, !"sign-return-address", i32 1}
// PART:  !{i32 8, !"sign-return-address", i32 1}
// BTE:   !{i32 8, !"sign-return-address", i32 0}
// B-KEY: !{i32 8, !"sign-return-address", i32 1}

// NONE:  !{i32 8, !"sign-return-address-all", i32 0}
// ALL:   !{i32 8, !"sign-return-address-all", i32 1}
// PART:  !{i32 8, !"sign-return-address-all", i32 0}
// BTE:   !{i32 8, !"sign-return-address-all", i32 0}
// B-KEY: !{i32 8, !"sign-return-address-all", i32 0}

// NONE:  !{i32 8, !"sign-return-address-with-bkey", i32 0}
// ALL:   !{i32 8, !"sign-return-address-with-bkey", i32 0}
// PART:  !{i32 8, !"sign-return-address-with-bkey", i32 0}
// BTE:   !{i32 8, !"sign-return-address-with-bkey", i32 0}
// B-KEY: !{i32 8, !"sign-return-address-with-bkey", i32 1}

void foo() {}
