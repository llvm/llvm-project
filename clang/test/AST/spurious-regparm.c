// RUN: %clang_cc1 -triple armv8.1m.main-eabi -mcmse -fsyntax-only %s -ast-dump | FileCheck %s
// REQUIRES: arm-registered-target
typedef int (*fn_t)(int) __attribute__((cmse_nonsecure_call));
// CHECK-NOT: regparm 0
