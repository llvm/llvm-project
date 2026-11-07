// RUN: %clang_cc1 -triple arm64-apple-ios   -coverage-data-file=/dev/null                                -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,OFF
// RUN: %clang_cc1 -triple arm64e-apple-ios  -coverage-data-file=/dev/null                                -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,OFF
// RUN: %clang_cc1 -triple aarch64-linux-gnu -coverage-data-file=/dev/null                                -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,OFF

// RUN: %clang_cc1 -triple arm64-apple-ios   -coverage-data-file=/dev/null -fptrauth-returns              -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,RETS \
// RUN:   --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening
// RUN: %clang_cc1 -triple aarch64-linux-gnu -coverage-data-file=/dev/null -fptrauth-returns              -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,RETS \
// RUN:   --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening

// RUN: %clang_cc1 -triple arm64-apple-ios   -coverage-data-file=/dev/null -fptrauth-auth-traps           -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,TRAPS \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening
// RUN: %clang_cc1 -triple aarch64-linux-gnu -coverage-data-file=/dev/null -fptrauth-auth-traps           -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,TRAPS \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-indirect-gotos --implicit-check-not=aarch64-jump-table-hardening

// RUN: %clang_cc1 -triple arm64-apple-ios   -coverage-data-file=/dev/null -fptrauth-indirect-gotos       -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,GOTOS \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=aarch64-jump-table-hardening
// RUN: %clang_cc1 -triple aarch64-linux-gnu -coverage-data-file=/dev/null -fptrauth-indirect-gotos       -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,GOTOS \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=aarch64-jump-table-hardening

// RUN: %clang_cc1 -triple arm64e-apple-ios  -coverage-data-file=/dev/null -faarch64-jump-table-hardening -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,JMPTBL \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos
// RUN: %clang_cc1 -triple aarch64-linux-gnu -coverage-data-file=/dev/null -faarch64-jump-table-hardening -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,JMPTBL \
// RUN:   --implicit-check-not=ptrauth-returns --implicit-check-not=ptrauth-auth-traps --implicit-check-not=ptrauth-indirect-gotos

// ALL: define internal void @__llvm_gcov_writeout() unnamed_addr #[[#T:]] {
// ALL: define internal void @__llvm_gcov_reset() unnamed_addr #[[#T]] {
// ALL: define internal void @__llvm_gcov_init() unnamed_addr #[[#T]] {

// RETS:      attributes #[[#T]] =
// RETS-SAME: "ptrauth-returns"
// RETS:      !llvm.module.flags = !{{{.*}}!3{{.*}}}
// RETS:      !3 = !{i32 7, !"ptrauth-returns", i32 1}

// TRAPS:      attributes #[[#T]] =
// TRAPS-SAME: "ptrauth-auth-traps"
// TRAPS:      !llvm.module.flags = !{{{.*}}!3{{.*}}}
// TRAPS:      !3 = !{i32 7, !"ptrauth-auth-traps", i32 1}

// GOTOS:      attributes #[[#T]] =
// GOTOS-SAME: "ptrauth-indirect-gotos"
// GOTOS:      !llvm.module.flags = !{{{.*}}!3{{.*}}}
// GOTOS:      !3 = !{i32 7, !"ptrauth-indirect-gotos", i32 1}

// JMPTBL:      attributes #[[#T]] =
// JMPTBL-SAME: "aarch64-jump-table-hardening"
// JMPTBL:      !llvm.module.flags = !{{{.*}}!3{{.*}}}
// JMPTBL:      !3 = !{i32 7, !"aarch64-jump-table-hardening", i32 1}

// OFF-NOT: attributes {{.*}} "ptrauth-
// OFF-NOT: !"ptrauth-returns"
// OFF-NOT: !"ptrauth-auth-traps"
// OFF-NOT: !"ptrauth-indirect-gotos"
// OFF-NOT: !"aarch64-jump-table-hardening"
