// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=NO-SKIP
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -mskip-rax-setup -emit-llvm %s -o - | FileCheck %s -check-prefix=SKIP

void f(void) {}

// SKIP: !"SkipRaxSetup", i32 1}
// NO-SKIP-NOT: "SkipRaxSetup"
