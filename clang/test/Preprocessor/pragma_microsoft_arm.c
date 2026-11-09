// RUN: %clang_cc1 -triple thumbv7-windows-msvc %s -fsyntax-only -verify -fms-extensions

// The compiler barriers are builtins on 32-bit Arm, as on x86 and AArch64.

// expected-no-diagnostics

#pragma intrinsic(_ReadBarrier)
#pragma intrinsic(_ReadWriteBarrier)
#pragma intrinsic(_WriteBarrier)
