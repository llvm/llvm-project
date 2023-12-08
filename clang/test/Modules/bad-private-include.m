// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules-cache-path=%t.cache -fmodules -fimplicit-module-maps -F %S/Inputs/bad-private-include %s -verify

// expected-no-diagnostics

@import Bad;
