// REQUIRES: aarch64-registered-target

// RUN: %clang -target aarch64-linux-pauthtest   %s -S -emit-llvm -o - 2>&1 | FileCheck --implicit-check-not=warning: %s
// RUN: %clang -target aarch64 -fptrauth-returns %s -S -emit-llvm -o - 2>&1 | FileCheck --implicit-check-not=warning: %s

/// Unsupported with pauthtest, warning emitted
__attribute__((target("branch-protection=pac-ret"))) void f1() {}
// CHECK:      warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=pac-ret"))) void f1() {}
__attribute__((target("branch-protection=gcs"))) void f2() {}
// CHECK:      warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=gcs"))) void f2() {}
__attribute__((target("branch-protection=standard"))) void f3() {}
// CHECK:      warning: unsupported 'branch-protection' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// CHECK-NEXT: __attribute__((target("branch-protection=standard"))) void f3() {}

/// Supported with pauthtest, no warning emitted
__attribute__((target("branch-protection=bti"))) void f4() {}

/// Supported with pauthtest, no warning emitted
__attribute__((target("branch-protection=none"))) void f5() {}

/// Check there are no branch protection function attributes which are unsupported with pauthtest

// CHECK-NOT:  attributes {{.*}} "sign-return-address"
// CHECK-NOT:  attributes {{.*}} "sign-return-address-key"
// CHECK-NOT:  attributes {{.*}} "branch-protection-pauth-lr"
// CHECK-NOT:  attributes {{.*}} "guarded-control-stack"

/// Check function attributes which are supported with pauthtest
// CHECK:      attributes {{.*}} "branch-target-enforcement"
