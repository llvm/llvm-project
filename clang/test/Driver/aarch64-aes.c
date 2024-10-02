// Test that +aes enables both FEAT_AES and FEAT_PMULL.
// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8-a+aes %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+aes"
// CHECK: "-target-feature" "+pmull"

// Test that +noaes disables both FEAT_AES and FEAT_PMULL.
// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8-a+aes+noaes %s 2>&1 | FileCheck %s --check-prefix=NOAES
// NOAES: "-target-feature" "-aes"
// NOAES: "-target-feature" "-pmull"
