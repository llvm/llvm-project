// REQUIRES: arm-registered-target

/// Check warning for
// RUN: %clang -target arm-arm-none-eabi %s -S -o - 2>&1 | FileCheck %s

__attribute__((target("branch-protection=bti"))) void f1() {}
__attribute__((target("branch-protection=pac-ret"))) void f2() {}
__attribute__((target("branch-protection=bti+pac-ret"))) void f3() {}
__attribute__((target("branch-protection=bti+pac-ret+leaf"))) void f4() {}

// CHECK: warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=bti"))) void f1() {}

// CHECK: warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=pac-ret"))) void f2() {}

// CHECK: warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=bti+pac-ret"))) void f3() {}

// CHECK: warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=bti+pac-ret+leaf"))) void f4() {}

/// Check there are no branch protection function attributes

// CHECK-NOT:  attributes { {{.*}} "sign-return-address"
// CHECK-NOT:  attributes { {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes { {{.*}} "branch-target-enforcement"
