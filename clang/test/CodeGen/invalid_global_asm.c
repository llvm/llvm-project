// REQUIRES: arm-registered-target
// RUN: not %clang_cc1 -emit-obj -triple armv6-unknown-unknown -o %t %s 2>&1 | FileCheck %s
#pragma clang diagnostic ignored "-Wmissing-noreturn"
__asm__(".Lfoo: movw r2, #:lower16:.Lbar - .Lfoo");
// CHECK: <inline asm>:1:8: error: instruction requires: armv6t2
