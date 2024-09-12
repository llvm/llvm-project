// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -emit-cir -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fno-common -emit-cir -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir %s -fcommon -emit-cir -o - | FileCheck %s -check-prefix=CHECK-COMMON

// CHECK-COMMON: cir.global common @x
// CHECK-DEFAULT: cir.global external @x
int x;

// CHECK-COMMON: cir.global external @ABC
// CHECK-DEFAULT: cir.global external @ABC
typedef void* (*fn_t)(long a, long b, char *f, int c);
fn_t ABC __attribute__ ((nocommon));

// CHECK-COMMON: cir.global common @y
// CHECK-DEFAULT: cir.global common @y
int y __attribute__((common));
