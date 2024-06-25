// RUN: %clang_cc1 -triple mips64-unknown-linux -O2 -target-abi n64 -emit-llvm %s -o - | FileCheck %s -check-prefix=N64
// RUN: %clang_cc1 -triple mips64-unknown-linux -O2 -target-abi n32 -emit-llvm %s -o - | FileCheck %s -check-prefix=N32
// RUN: %clang_cc1 -triple mips-unknown-linux -O2 -target-abi o32 -emit-llvm %s -o - | FileCheck %s -check-prefix=O32

#include <stdarg.h>

unsigned foo(int x, ...) {
  va_list valist;
  va_start(valist, x);
  unsigned a;
  a = va_arg(valist, unsigned);
  return a;
}

void foo1(void) {
  unsigned f = 0xffffffe0;
  foo(1,f);
}

//N64: call signext i32 (i32, ...) @foo(i32 signext poison, i32 noundef signext -32)
//N32: call signext i32 (i32, ...) @foo(i32 signext poison, i32 noundef signext -32)
//O32: call i32 (i32, ...) @foo(i32 signext poison, i32 noundef signext -32)
