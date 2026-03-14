// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -triple aarch64-unknown-unknown -target-feature +neon -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -verify %s
// expected-no-diagnostics
// REQUIRES: aarch64-registered-target
@import _Builtin_intrinsics.arm;
@import _Builtin_intrinsics.arm.neon;
