// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - -mcmodel=medium | FileCheck %s --check-prefix=IR-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - -mcmodel=medium -mlarge-data-threshold=200 | FileCheck %s --check-prefix=IR-CUSTOM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - -mcmodel=large -mlarge-data-threshold=200 | FileCheck %s --check-prefix=IR-CUSTOM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - -mcmodel=small -mlarge-data-threshold=200 | FileCheck %s --check-prefix=IR-SMALL
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o - -mcmodel=medium -mlarge-data-threshold=200 | FileCheck %s --check-prefix=ASM-SMALL
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -S %s -o - -mcmodel=medium -mlarge-data-threshold=2 | FileCheck %s --check-prefix=ASM-LARGE

// IR-DEFAULT: !{i32 1, !"Large Data Threshold", i64 0}
// IR-CUSTOM: !{i32 1, !"Large Data Threshold", i64 200}
// IR-SMALL-NOT: !"Large Data Threshold"

// ASM-SMALL-NOT: movabsq
// ASM-LARGE: movabsq

static int i;

int f() {
    return i;
}

