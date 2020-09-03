// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 %s -fno-common -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 %s -fcommon -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-COMMON

// CHECK-COMMON: @x = common {{(dso_local )?}}global
// CHECK-DEFAULT: @x = {{(dso_local )?}}global
int x;

// CHECK-COMMON: @ABC = {{(dso_local )?}}global
// CHECK-DEFAULT: @ABC = {{(dso_local )?}}global
typedef void* (*fn_t)(long a, long b, char *f, int c);
fn_t ABC __attribute__ ((nocommon));

// CHECK-COMMON: @y = common {{(dso_local )?}}global
// CHECK-DEFAULT: @y = common {{(dso_local )?}}global
int y __attribute__((common));
