// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 3 -emit-llvm %s -o - | FileCheck %s -check-prefix=HARD
// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 3 -mfloat-abi soft -emit-llvm %s -o - | FileCheck %s -check-prefix=SOFT

// HARD: define{{.*}} void @f1(float %a)
// SOFT: define{{.*}} void @f1(float inreg %a)
void f1(float a) {}
