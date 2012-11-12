// REQUIRES: ppc32-registered-target
// RUN: %clang_cc1 -triple powerpc-apple-darwin -emit-llvm -o - %s| FileCheck -check-prefix=DARWINPPC-CHECK %s

int boolsize = sizeof(_Bool);
//DARWINPPC-CHECK: boolsize = global i32 4, align 4

