// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 %s -fno-common -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-NOCOMMON

// CHECK-DEFAULT: @x = common {{(dso_local )?}}global
// CHECK-NOCOMMON: @x = {{(dso_local )?}}global
int x;

// CHECK-DEFAULT: @ABC = {{(dso_local )?}}global
// CHECK-NOCOMMON: @ABC = {{(dso_local )?}}global
typedef void* (*fn_t)(long a, long b, char *f, int c);
fn_t ABC __attribute__ ((nocommon));

// CHECK-DEFAULT: @y = common {{(dso_local )?}}global
// CHECK-NOCOMMON: @y = common {{(dso_local )?}}global
int y __attribute__((common));
