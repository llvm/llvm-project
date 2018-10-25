// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -msign-return-address=none  %s | FileCheck %s --check-prefix=CHECK --check-prefix=NONE
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.2-a -S -emit-llvm -o - -msign-return-address=all  %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL --check-prefix=A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -msign-return-address=all %s   | FileCheck %s --check-prefix=CHECK --check-prefix=ALL --check-prefix=A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -msign-return-address=non-leaf %s | FileCheck %s --check-prefix=CHECK --check-prefix=PARTIAL --check-prefix=A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -msign-return-address=all %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL --check-prefix=A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.4-a -S -emit-llvm -o - -msign-return-address=all %s | FileCheck %s --check-prefix=CHECK --check-prefix=ALL --check-prefix=A-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key %s   | FileCheck %s --check-prefix=CHECK --check-prefix=PARTIAL --check-prefix=B-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -mbranch-protection=pac-ret+b-key+leaf %s   | FileCheck %s --check-prefix=CHECK --check-prefix=ALL --check-prefix=B-KEY
// RUN: %clang -target aarch64-arm-none-eabi -march=armv8.3-a -S -emit-llvm -o - -mbranch-protection=bti %s | FileCheck %s --check-prefix=CHECK --check-prefix=BTE

// REQUIRES: aarch64-registered-target

// CHECK: @foo() #[[ATTR:[0-9]*]]
//
// NONE-NOT: "sign-return-address"={{.*}}

// PARTIAL: "sign-return-address"="non-leaf"

// ALL: "sign-return-address"="all"

// BTE: "branch-target-enforcement"

// A-KEY: "sign-return-address-key"="a_key"

// B-KEY: "sign-return-address-key"="b_key"

void foo() {}
