// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -S -o - %s | \
// RUN:   FileCheck %s --check-prefix=ATEXIT --implicit-check-not=__mod_term_func
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -ffreestanding -S -o - %s | \
// RUN:   FileCheck %s --check-prefix=MODTERM --implicit-check-not=___cxa_atexit

// Freestanding MachO modules are marked so that their global destructors are
// not lowered to __cxa_atexit() calls, even after LTO.
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -ffreestanding -emit-llvm \
// RUN:   -o - %s | FileCheck %s --check-prefix=FLAG
// RUN: %clang_cc1 -triple x86_64-apple-macos11 -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=NOFLAG
// RUN: %clang_cc1 -triple x86_64-unknown-linux -ffreestanding -emit-llvm \
// RUN:   -o - %s | FileCheck %s --check-prefix=NOFLAG

// ATEXIT: callq ___cxa_atexit

// MODTERM:      .section __DATA,__mod_term_func,mod_term_funcs
// MODTERM-NEXT: .p2align 3
// MODTERM-NEXT: .quad _dtor

// FLAG:       !{i32 7, !"disable-atexit-based-global-dtor-lowering", i32 1}
// NOFLAG-NOT: disable-atexit-based-global-dtor-lowering

void f(void);
__attribute__((destructor)) void dtor(void) { f(); }
