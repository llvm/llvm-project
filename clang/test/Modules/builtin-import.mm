// RUN: rm -rf %t
// RUN: %clang -cc1 -fsyntax-only -nobuiltininc -nostdinc++ -isysroot %S/Inputs/libc-libcxx/sysroot -isystem %S/Inputs/libc-libcxx/sysroot/usr/include/c++/v1 -isystem %S/Inputs/libc-libcxx/sysroot/usr/include -std=c++11 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c++ -fmodules-local-submodule-visibility %s

#include <stdio.h>
#include <stddef.h>
#include <cstddef>

typedef ptrdiff_t try1_ptrdiff_t;
typedef my_ptrdiff_t try2_ptrdiff_t;

