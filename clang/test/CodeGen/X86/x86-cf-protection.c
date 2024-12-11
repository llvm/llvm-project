// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=return %s | FileCheck %s --check-prefix=RETURN
// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=branch %s | FileCheck %s --check-prefix=BRANCH
// RUN: %clang_cc1 -E -triple i386 -dM -o - -fcf-protection=full %s   | FileCheck %s --check-prefix=FULL
// RUN: %clang_cc1 -E -triple=x86_64 -dM -o - -fcf-protection=none %s | FileCheck %s --check-prefix=NOTCET
// RUN: not %clang_cc1 -emit-llvm-only -triple i386 -target-cpu pentium-mmx -fcf-protection=branch %s 2>&1 | FileCheck %s --check-prefix=NOCFPROT
// RUN: %clang_cc1 -triple=x86_64 -o - -fcf-protection=return %s -emit-llvm | FileCheck %s --check-prefixes=CFPROTR,CFPROTNONE
// RUN: %clang_cc1 -triple=x86_64 -o - -fcf-protection=branch %s -emit-llvm | FileCheck %s --check-prefixes=CFPROTB,CFPROTNONE
// RUN: %clang_cc1 -triple=x86_64 -o - -fcf-protection=full %s -emit-llvm | FileCheck %s --check-prefixes=CFPROTR,CFPROTB,CFPROTNONE
// RUN: %clang_cc1 -triple=x86_64 -o - -fcf-protection=none %s -emit-llvm | FileCheck %s --check-prefixes=CFPROTNONE

// RETURN: #define __CET__ 2
// BRANCH: #define __CET__ 1
// FULL: #define __CET__ 3
// NOTCET-NOT: #define __CET__
// NOCFPROT: error: option 'cf-protection=branch' cannot be specified on this target
// CFPROTR: !{i32 8, !"cf-protection-return", i32 1}
// CFPROTB: !{i32 8, !"cf-protection-branch", i32 1}
// CFPROTNONE-NOT: cf-protection-

void foo() {}
