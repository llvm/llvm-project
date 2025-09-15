// RUN: %clang_cc1 -triple arm64-windows-msvc -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

int simple_int(va_list ap) {
// CHECK-LABEL: define dso_local i32 @simple_int
  return va_arg(ap, int);
// CHECK: [[RESULT:%[a-z_0-9]+]] = load i32, ptr %argp.cur
// CHECK: ret i32 [[RESULT]]
}
