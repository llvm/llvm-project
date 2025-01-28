// RUN: %clang_cc1 -triple arm64-apple-ios                    -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,OFF
// RUN: %clang_cc1 -triple arm64e-apple-ios                   -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,OFF
// RUN: %clang_cc1 -triple aarch64-linux-gnu                  -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,OFF

// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-calls  -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,CALLS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-calls  -emit-llvm %s  -o - | FileCheck %s --check-prefixes=ALL,CALLS

// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-returns -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,RETS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-returns -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,RETS

// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-auth-traps -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,TRAPS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-auth-traps -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,TRAPS

// RUN: %clang_cc1 -triple arm64-apple-ios   -fptrauth-indirect-gotos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,GOTOS
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fptrauth-indirect-gotos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,GOTOS

// RUN: %clang_cc1 -triple arm64e-apple-ios  -faarch64-jump-table-hardening -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,JMPTBL
// RUN: %clang_cc1 -triple aarch64-linux-gnu -faarch64-jump-table-hardening -emit-llvm %s -o - | FileCheck %s --check-prefixes=ALL,JMPTBL

// ALL: define {{(dso_local )?}}void @test() #0
void test() {
}

// CALLS: attributes #0 = {{{.*}} "ptrauth-calls" {{.*}}}

// RETS: attributes #0 = {{{.*}} "ptrauth-returns" {{.*}}}

// TRAPS: attributes #0 = {{{.*}} "ptrauth-auth-traps" {{.*}}}

// GOTOS: attributes #0 = {{{.*}} "ptrauth-indirect-gotos" {{.*}}}

// JMPTBL: attributes #0 = {{{.*}} "aarch64-jump-table-hardening" {{.*}}}

// OFF-NOT: attributes {{.*}} "ptrauth-
