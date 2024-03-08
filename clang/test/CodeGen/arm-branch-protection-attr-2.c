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

// NONE-NOT: !"branch-target-enforcement"
// PART-NOT: !"branch-target-enforcement"
// ALL-NOT:  !"branch-target-enforcement"
// BTE:      !{i32 8, !"branch-target-enforcement", i32 2}

// NONE-NOT: !"sign-return-address"
// PART:     !{i32 8, !"sign-return-address", i32 2}
// ALL:      !{i32 8, !"sign-return-address", i32 2}
// BTE-NOT:  !"sign-return-address"

// NONE-NOT: !"sign-return-address-all", i32 0}
// PART-NOT: !"sign-return-address-all", i32 0}
// ALL:      !{i32 8, !"sign-return-address-all", i32 2}
// BTE-NOT:  !"sign-return-address-all", i32 0}

void foo() {}
