// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=return %s | FileCheck %s --check-prefix=RETURN
// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=branch %s | FileCheck %s --check-prefix=BRANCH
// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=full %s   | FileCheck %s --check-prefix=FULL
// RUN: not %clang_cc1 -emit-llvm-only -triple i386 -target-cpu pentium-mmx -fcf-protection=branch %s 2>&1 | FileCheck %s --check-prefix=NOCFPROT

// RETURN: #define __CET__ 2
// BRANCH: #define __CET__ 1
// FULL: #define __CET__ 3
// CFPROT: !{i32 8, !"cf-protection-branch", i32 1}

// NOCFPROT: error: option 'cf-protection=branch' cannot be specified on this target

void foo() {}
