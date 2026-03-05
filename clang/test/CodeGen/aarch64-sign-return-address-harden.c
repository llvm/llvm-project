// RUN: %clang -target aarch64 -S -emit-llvm -o - %s -mbranch-protection=none \
// RUN: | FileCheck %s --check-prefixes=CHECK-NO-HARDEN
// RUN: %clang -target aarch64 -S -emit-llvm -o - %s -mbranch-protection=pac-ret \
// RUN: | FileCheck %s --check-prefixes=CHECK-NO-HARDEN
// RUN: %clang -target aarch64 -S -emit-llvm -o - %s -mbranch-protection=pac-ret -mharden-pac-ret=none \
// RUN: | FileCheck %s --check-prefixes=CHECK-NO-HARDEN
// RUN: %clang -target aarch64 -S -emit-llvm -o - %s -mbranch-protection=pac-ret -mharden-pac-ret=load-return-address \
// RUN: | FileCheck %s --check-prefixes=CHECK-HARDEN

void foo() {}

// CHECK-NO-HARDEN-NOT: attributes #0 = {{.*}}"sign-return-address-harden"
// CHECK-HARDEN:        attributes #0 = {{.*}}"sign-return-address-harden"="load-return-address"