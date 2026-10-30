// Under -mcmodel=large the interceptor can be more than 2 GiB away from the
// trampoline, so the trampoline must jump through the GOT.
// RUN: %clangxx -S -mcmodel=large -I%crt_src/lib %s -o - | FileCheck %s
//
// REQUIRES: x86_64-target-arch

#include "interception/interception.h"

// CHECK-LABEL: __interceptor_trampoline_foo:
// CHECK-NOT: jmp
// CHECK: jmpq *__interceptor_foo@GOTPCREL(%rip)
INTERCEPTOR(int, foo, int x) { return REAL(foo)(x); }
