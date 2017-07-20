// RUN: %clang_cc1 -triple arm64-windows-msvc -emit-llvm -o - %s | FileCheck --check-prefix=CHECK %s

#include <stdarg.h>

int simple_int(va_list ap) {
// CHECK-LABEL: define i32 @simple_int
  return va_arg(ap, int);
// CHECK: [[ADDR:%[a-z_0-9]+]] = bitcast i8* %argp.cur to i32*
// CHECK: [[RESULT:%[a-z_0-9]+]] = load i32, i32* [[ADDR]]
// CHECK: ret i32 [[RESULT]]
}
