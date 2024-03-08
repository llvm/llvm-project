// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -msign-return-address=none     %s | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -msign-return-address=all      %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL
// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -msign-return-address=non-leaf %s | FileCheck %s --check-prefix=CHECK --check-prefix=PART

// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -mbranch-protection=none %s          | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -mbranch-protection=pac-ret+leaf  %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL
// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key %s | FileCheck %s --check-prefix=CHECK --check-prefix=B-KEY
// RUN: %clang -target aarch64-none-elf -S -emit-llvm -o - -mbranch-protection=bti %s           | FileCheck %s --check-prefix=CHECK --check-prefix=BTE

// REQUIRES: aarch64-registered-target

// Check there are no branch protection function attributes

// CHECK-LABEL: @foo() #[[#ATTR:]]

// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"

// Check module attributes

// NONE-NOT:  !"branch-target-enforcement"
// ALL-NOT:   !"branch-target-enforcement"
// PART-NOT:  !"branch-target-enforcement"
// BTE:       !{i32 8, !"branch-target-enforcement", i32 2}
// B-KEY-NOT: !"branch-target-enforcement"

// NONE-NOT:  !"sign-return-address"
// ALL:   !{i32 8, !"sign-return-address", i32 2}
// PART:  !{i32 8, !"sign-return-address", i32 2}
// BTE-NOT:   !"sign-return-address"
// B-KEY: !{i32 8, !"sign-return-address", i32 2}

// NONE-NOT:  !"sign-return-address-all"
// ALL:   !{i32 8, !"sign-return-address-all", i32 2}
// PART-NOT:  !"sign-return-address-all"
// BTE-NOT:   !"sign-return-address-all"
// B-KEY-NOT: !"sign-return-address-all"

// NONE-NOT:  !"sign-return-address-with-bkey"
// ALL-NOT:   !"sign-return-address-with-bkey"
// PART-NOT:  !"sign-return-address-with-bkey"
// BTE-NOT:   !"sign-return-address-with-bkey"
// B-KEY: !{i32 8, !"sign-return-address-with-bkey", i32 2}

void foo() {}
