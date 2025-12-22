// REQUIRES: arm-registered-target

/// Check warning for
// RUN: %clang -target arm-arm-none-eabi -march=armv7-m -mbranch-protection=bti %s -S -emit-llvm -o - 2>&1 | FileCheck %s

__attribute__((target("arch=cortex-m0"))) void f() {}

// CHECK: warning: ignoring the 'branch-protection' attribute because the 'cortex-m0' architecture does not support it [-Wbranch-protection]
// CHECK-NEXT: __attribute__((target("arch=cortex-m0"))) void f() {}

/// Check there are no branch protection function attributes

// CHECK-NOT:  attributes { {{.*}} "sign-return-address"
// CHECK-NOT:  attributes { {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes { {{.*}} "branch-target-enforcement"

/// Check that there are branch protection module attributes despite the warning.
// CHECK: !{i32 8, !"branch-target-enforcement", i32 2}
