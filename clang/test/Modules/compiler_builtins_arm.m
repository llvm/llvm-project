// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -triple thumbv7-none-linux-gnueabihf -target-abi aapcs -target-cpu cortex-a8 -mfloat-abi hard -std=c99 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -D__need_wint_t %s -verify
// expected-no-diagnostics
// REQUIRES: arm-registered-target
@import _Builtin_intrinsics.arm.neon;
