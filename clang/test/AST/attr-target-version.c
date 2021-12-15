// RUN: %clang_cc1 -triple aarch64-linux-gnu -ast-dump %s | FileCheck %s

int __attribute__((target_version("sve2-bitperm + sha2"))) foov(void) { return 1; }
int __attribute__((target_clones(" lse + fp + sha3 "))) fooc(void) { return 2; }
// CHECK: TargetVersionAttr
// CHECK: sve2-bitperm + sha2
// CHECK: TargetClonesAttr
// CHECK: fp+lse+sha3 default
