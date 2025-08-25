// RUN: %clang_cc1 -emit-llvm %s -o - -fms-extensions -triple=x86_64-pc-win32 -foverflow-behavior-types | FileCheck %s

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

typedef int __wrap int_wrap;

// CHECK: define dso_local void @"?test_wrap_int@@YAXU?$ObtWrap_@H@__clang@@@Z"
void test_wrap_int(int_wrap x) {}

// CHECK: define dso_local void @"?test_no_wrap_int@@YAXU?$ObtNoWrap_@H@__clang@@@Z"
void test_no_wrap_int(int __no_wrap y) {}
