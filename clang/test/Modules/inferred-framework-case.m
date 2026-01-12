// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify -DA
// FIXME: PR20299 - getCanonicalName() is not implemented on Windows.
// UNSUPPORTED: system-windows

@import MOdule; // expected-error{{module 'MOdule' not found}}
@import Module;
