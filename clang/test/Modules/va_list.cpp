// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/va_list %s -verify
// expected-no-diagnostics

@import left;
@import right;

void g(int k, ...) { f<int>(k); }
