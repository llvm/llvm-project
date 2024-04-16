// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -O1 -emit-llvm -o - %s | opt --passes='module(expand-variadics,inline)' -S | FileCheck %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu -O1 -emit-llvm -o - %s | opt --passes='module(expand-variadics,inline)' -S | FileCheck %s

// neither arm arch is implemented yet, leaving it here as a reminder
// armv6 is a ptr as far as the struct is concerned, but possibly also a [1 x i32] passed by value
// that seems insistent, maybe leave 32 bit arm alone for now
// aarch64 is a struct of five things passed using byval memcpy

// R-N: %clang_cc1 -triple=armv6-none--eabi -O1 -emit-llvm -o - %s | opt --passes=expand-variadics -S | FileCheck %s
// R-N: %clang_cc1 -triple=aarch64-none-linux-gnu -O1 -emit-llvm -o - %s | opt --passes=expand-variadics -S | FileCheck %s



// expand-variadics rewrites calls to variadic functions into calls to
// equivalent functions that take a va_list argument. A property of the
// implementation is that said "equivalent function" may be a pre-existing one.
// This is equivalent to inlining a sufficiently simple variadic wrapper.

#include <stdarg.h>

typedef int FILE; // close enough for this test

// fprintf is sometimes implemented as a call to vfprintf. That fits the
// pattern the transform pass recognises - given an implementation of fprintf
// in the IR module, calls to it can be rewritten into calls into vfprintf.

// CHECK-LABEL: define{{.*}} i32 @fprintf(
// CHECK-LABEL: define{{.*}} i32 @call_fprintf(
// CHECK-NOT:   @fprintf
// CHECK:       @vfprintf
int vfprintf(FILE *restrict f, const char *restrict fmt, va_list ap);
int fprintf(FILE *restrict f, const char *restrict fmt, ...)
{
  int ret;
  va_list ap;
  va_start(ap, fmt);
  ret = vfprintf(f, fmt, ap);
  va_end(ap);
  return ret;
}
int call_fprintf(FILE *f)
{
  int x = 42;
  double y = 3.14;
  return fprintf(f, "int %d dbl %g\n", x, y);
}

// Void return type is also OK

// CHECK-LABEL: define{{.*}} void @no_result(
// CHECK-LABEL: define{{.*}} void @call_no_result(
// CHECK-NOT:   @no_result
// CHECK:       @vno_result
void vno_result(const char * fmt, va_list);
void no_result(const char * fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vno_result(fmt, ap);
  va_end(ap);
}
void call_no_result(FILE *f)
{
  int x = 101;
  no_result("", x);
}

// The vaend in the forwarding implementation is optional where it's a no-op

// CHECK-LABEL: define{{.*}} i32 @no_vaend(
// CHECK-LABEL: define{{.*}} i32 @call_no_vaend(
// CHECK-NOT:   @no_vaend
// CHECK:       @vno_vaend
int vno_vaend(int x, va_list);
int no_vaend(int x, ...)
{
  va_list ap;
  va_start(ap, x);
  return vno_vaend(x, ap);
}
int call_no_vaend(int x)
{
  return no_vaend(x, 10, 2.5f);
}
