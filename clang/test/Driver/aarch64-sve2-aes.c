// Test that +sve2-aes enables both FEAT_SVE_AES and FEAT_SVE_PMULL128.
// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8-a+sve2-aes %s 2>&1 | FileCheck %s
// CHECK: "-target-feature" "+sve2-aes"
// CHECK: "-target-feature" "+sve2-pmull128"

// Test that +nosve2-aes disables both FEAT_SVE_AES and FEAT_SVE_PMULL128.
// RUN: %clang -### --target=aarch64-linux-gnu -march=armv8-a+sve2-aes+nosve2-aes %s 2>&1 | FileCheck %s --check-prefix=NOSVE2-AES
// NOSVE2-AES: "-target-feature" "-sve2-aes"
// NOSVE2-AES: "-target-feature" "-sve2-pmull128"
