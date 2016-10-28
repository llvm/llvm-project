// REQUIRES: system-darwin

// RUN: rm -rf %t
// RUN: %clang -cc1 -fsyntax-only -nostdinc++ -isysroot %S/Inputs/libc-libcxx/sysroot -isystem %S/Inputs/libc-libcxx/sysroot/usr/include/c++/v1 -F%S/Inputs/libc-libcxx/sysroot/Frameworks -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c++ %s

#include <A/A.h>
