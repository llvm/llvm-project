// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/libc-libcxx/include/c++ -I %S/Inputs/libc-libcxx/include %s -verify
// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility -nostdinc++ -isystem %S/Inputs/libc-libcxx/sysroot/usr/include/c++/v1 -isystem %S/Inputs/libc-libcxx/sysroot/usr/include -fsyntax-only %s -verify
// expected-no-diagnostics

#include "math.h"

int n = abs(0);
float f = abs<float>(0.f);
