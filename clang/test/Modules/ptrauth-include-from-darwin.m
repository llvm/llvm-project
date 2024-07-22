// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/ptrauth-include-from-darwin %s -verify
// expected-no-diagnostics

@import libc;
void bar() { foo(); }
