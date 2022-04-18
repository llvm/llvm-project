// REQUIRES: arm-registered-target
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -S -emit-llvm -o - -mbranch-protection=none %s          | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -S -emit-llvm -o - -mbranch-protection=pac-ret       %s | FileCheck %s --check-prefix=CHECK --check-prefix=PART
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -S -emit-llvm -o - -mbranch-protection=pac-ret+leaf  %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key %s | FileCheck %s --check-prefix=CHECK --check-prefix=PART
// RUN: %clang -target arm-arm-none-eabi -march=armv8.1-m.main -S -emit-llvm -o - -mbranch-protection=bti %s           | FileCheck %s --check-prefix=CHECK --check-prefix=BTE

// Check there are no branch protection function attributes

// CHECK-LABEL: @foo() #[[#ATTR:]]

// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes #[[#ATTR]] = { {{.*}} "branch-target-enforcement"

// Check module attributes

// NONE:  !{i32 8, !"branch-target-enforcement", i32 0}
// PART:  !{i32 8, !"branch-target-enforcement", i32 0}
// ALL:   !{i32 8, !"branch-target-enforcement", i32 0}
// BTE:   !{i32 8, !"branch-target-enforcement", i32 1}

// NONE:  !{i32 8, !"sign-return-address", i32 0}
// PART:  !{i32 8, !"sign-return-address", i32 1}
// ALL:   !{i32 8, !"sign-return-address", i32 1}
// BTE:   !{i32 8, !"sign-return-address", i32 0}

// NONE:  !{i32 8, !"sign-return-address-all", i32 0}
// PART:  !{i32 8, !"sign-return-address-all", i32 0}
// ALL:   !{i32 8, !"sign-return-address-all", i32 1}
// BTE:   !{i32 8, !"sign-return-address-all", i32 0}

void foo() {}
