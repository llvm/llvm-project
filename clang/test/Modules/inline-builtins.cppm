// REQUIRES: !system-windows
//
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -O3 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -O3 %t/test.cc -fmodule-file=a=%t/a.pcm \
// RUN:   -emit-llvm -o - | FileCheck %t/test.cc

//--- memmove.h
typedef long unsigned int size_t;
extern "C" void *memmove (void *__dest, const void *__src, size_t __n)
     throw () __attribute__ ((__nonnull__ (1, 2)));
extern "C" __inline __attribute__ ((__always_inline__)) __attribute__ ((__gnu_inline__)) void *
 memmove (void *__dest, const void *__src, size_t __len) throw ()
{
  return __builtin_memmove(__dest, __src, __len);
}

//--- a.cppm
module;
#include "memmove.h"
export module a;
export using ::memmove;

//--- test.cc
import a;

void test() {
  int a, b;
  unsigned c = 0;
  memmove(&a, &b, c);
}

// CHECK-NOT: memmove
